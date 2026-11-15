import re
from re import Pattern
import torch
from torch import Tensor
import torch.nn as nn
from typing import Iterable, Union
from ._hook_handler import _MetaPEFTHandler
from typing import Union, List, Callable, TypeVar

_RegEx = Union[str, Pattern, List[Union[str, Pattern]]]
_Tensor = TypeVar('_Tensor', bound=Tensor)
_Module = TypeVar('_Module', bound=torch.nn.Module)
_Param = TypeVar('_Param', bound=torch.nn.Parameter)

_PreForwardHook = Callable[[_Module, _Tensor], None]
_ForwardHook = Callable[[_Module, _Tensor, _Tensor], None]
_BackwardHook = Callable[[_Module, _Tensor], None]


class MultiPoleLoRA(nn.Module):
    def __init__(
        self,
        layer: nn.Linear,
        rank: int,
        alpha: int = None,
        n_pole: int = 1,
        name: Union[str, Iterable[str]] = None,
    ):
        """
        Linear Rank Adaptation (LoRA) layer.
        Args:
            layer (nn.Linear):
                The linear layer to apply LoRA.

            rank (int):
                The rank of the LoRA.

            alpha (int):
                The alpha value of the LoRA. (default: rank * 2)

            name (Union[str, Iterable[str]]):
                The name of the LoRA. (default: 'base')

        The LoRA layer is activated by default for the first LoRA.
        You can activate() or deactivate() the LoRA
        It is recommended to get the LoRAHandle by use the apply_lora().
        """
        super().__init__()
        self.rank = rank
        self.dim_in = layer.weight.shape[1]
        self.dim_out = layer.weight.shape[0]
        self.alpha = alpha if alpha is not None else rank
        self.lora_scale = self.alpha / self.rank
        self.n_pole = n_pole

        self.register_buffer(
            "W", layer.weight.detach().clone().reshape(
                1, self.dim_out, self.dim_in)
        )
        if layer.bias is not None:
            self.register_buffer(
                "b", layer.bias.detach().clone().reshape(
                    self.dim_out)
            )
        else:
            self.b = None

        # If name is None, set the name as 'base'.
        if name is None:
            name = ("base",)
        if isinstance(name, str):
            name = (name,)
        self._as = nn.ParameterDict()
        self._bs = nn.ParameterDict()
        self._zs = nn.ParameterDict()
        for n in name:
            self.add_lora(n)
        # LoRA is activated by default for the first LoRA.
        self._active = False
        self.activate(name[0])

    def reset_parameters(self) -> None:
        for (_na, _a), (_nb, _b) in zip(self._as.items(), self._bs.items()):
            nn.init.normal_(_a, 0, 1)
            nn.init.zeros_(_b)

    def add_lora(self, name: str, active_init: bool = False):
        self._zs[name] = nn.Parameter(
            torch.zeros(
                self.n_pole, 1, self.dim_in, requires_grad=True)
        )
        self._as[name] = nn.Parameter(
            torch.randn(
                self.n_pole, self.dim_in,  self.rank, requires_grad=True)
        )
        self._bs[name] = nn.Parameter(
            torch.zeros(
                self.n_pole, self.dim_out, self.rank, requires_grad=True)
        )
        if active_init:
            self._lora_a = self._as[name]
            self._lora_b = self._bs[name]

    def remove_lora(self, name: str):
        if name in self._as.keys():
            del self._as[name]
        if name in self._bs.keys():
            del self._bs[name]
        if name in self._zs.keys():
            del self._zs[name]

    def _active_forward(self, x):
        shape = x.shape
        nvec = torch.tensor(shape).cumprod(0)[-2]
        _x = x.reshape(1, nvec, self.dim_in)

        _z = _x + self._lora_z  # (n_pole, nvec, dim_in)
        # _Wz = _z @ self.W.transpose(-1, -2)  # (n_pole, nvec, out)
        # _Az = _z @ self._lora_a.transpose(-1, -2)  # (n_pole, nvec, rank)
        # _BAz = _Az @ self._lora_b.transpose(-1, -2)
        # _BAz_plus_Wz = _Wz + _BAz  # (n_pole, nvec, out)
        # _BAz_plus_Wz = nn.functional.tanh(_BAz_plus_Wz)
        # _BAz_plus_Wz = _BAz_plus_Wz.mean(dim=0)  # (nvec, out)

        # # (n_pole, in, out)
        _BA = self._lora_b @ self._lora_a.transpose(-1, -2)
        _BA_plus_W = self.W + _BA  # (n_pole, out, in)
        _BA_plus_Wz = _z @ _BA_plus_W.transpose(-1, -2)
        _BA_plus_Wz = nn.functional.tanh(
            _BA_plus_Wz
        ).mean(dim=0)  # (nvec, out)
        if self.b is not None:
            _BAz_plus_Wz = _BA_plus_Wz + self.b  # (n_pole, nvec, out)
        _BAz_plus_Wz = _BAz_plus_Wz.reshape(
            shape[:-1] + (self.dim_out,))  # (batch, out)
        return _BAz_plus_Wz

        # # (n_pole, nvec, out)
        # BAz_plus_Wz = _z @ BA_plus_W.transpose(-1, -2)
        # BAz_plus_Wz = BAz_plus_Wz.mean(dim=0)  # (nvec, out)
        # if self.b is not None:
        #     BAz_plus_Wz = BAz_plus_Wz + self.b
        # BAz_plus_Wz = BAz_plus_Wz.reshape(
        #     shape[:-1] + (self.dim_out,))  # (batch, out)
        # return BAz_plus_Wz

    def _inactive_forward(self, x):
        if self.b is not None:
            return x @ self.W.T + self.b
        return x @ self.W.T

    def forward(self, x):
        if self._active:
            return self._active_forward(x)
        else:
            return self._inactive_forward(x)

    def activate(self, name=None):
        if name is None:
            name = "base"
        self._lora_a = self._as[name]
        self._lora_b = self._bs[name]
        self._lora_z = self._zs[name]
        self._active = True
        self._active_name = name

    def deactivate(self):
        self._active = False
        self._active_name = None

    def extra_repr(self):
        _repr = ""
        _repr += f"in_features={self.dim_in},"
        _repr += f"out_features={self.dim_out}, "
        _repr += f"rank={self.rank}, alpha={self.alpha}"
        if self._active:
            _repr += f", active={self._active_name}, "
            _repr += f" mode={list(self._as.keys())}"
        return _repr


class MultiPoleLoRAHandler(_MetaPEFTHandler):
    def __init__(
            self,
            module,
            rank,
            alpha=None,
            n_pole=3,
            module_names=[],
            lora_names=['base',],
            **kwargs
    ):
        super().__init__(module, **kwargs)
        self._handle = []
        for name, sub_module in self.module.named_modules():
            if re.match("|".join(module_names), name) and isinstance(sub_module, nn.Linear):
                lora = MultiPoleLoRA(
                    sub_module, rank, alpha=alpha, n_pole=n_pole, name=lora_names)
                self._handle.append(lora)
                _name = name.split(".")
                _module = self.module
                for i in range(len(_name) - 1):
                    _module = _module.__getattr__(_name[i])
                _module.__setattr__(_name[-1], lora)

    def enable(self):
        for handle in self._handle:
            handle.activate()

    def disable(self):
        for handle in self._handle:
            handle.deactivate()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
