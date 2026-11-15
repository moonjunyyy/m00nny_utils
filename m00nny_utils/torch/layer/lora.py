import torch
import torch.nn as nn
from typing import Iterable, Union


class LoRA(nn.Module):
    def __init__(
        self,
        layer: nn.Linear,
        rank: int,
        alpha: int = None,
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
        self.register_buffer("W", layer.weight.detach().clone())
        if layer.bias is not None:
            self.register_buffer("b", layer.bias.detach().clone())
        else:
            self.b = None
        self.rank = rank
        self.dim_in = layer.weight.shape[1]
        self.dim_out = layer.weight.shape[0]
        self.alpha = alpha if alpha is not None else rank
        self.lora_scale = self.alpha / self.rank

        # If name is None, set the name as 'base'.
        if name is None:
            name = ("base",)
        if isinstance(name, str):
            name = (name,)
        self._as = nn.ParameterDict()
        self._bs = nn.ParameterDict()
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
        self._as[name] = nn.Parameter(
            torch.randn(self.dim_in, self.rank, requires_grad=True)
        )
        self._bs[name] = nn.Parameter(
            torch.zeros(self.dim_out, self.rank, requires_grad=True)
        )
        if active_init:
            self._lora_a = self._as[name]
            self._lora_b = self._bs[name]

    def remove_lora(self, name: str):
        if name in self._as.keys():
            del self._as[name]
        if name in self._bs.keys():
            del self._bs[name]

    def _active_forward(self, x):
        # (out, rank) @ (rank, in) = (out, in)
        BA = self._lora_b @ self._lora_a.T
        # (out, in)
        BA_plus_W = self.W + BA * self.lora_scale
        # (batch, in) @ (in, out) = (batch, out)
        BAx_plus_Wx = x @ BA_plus_W.T
        if self.b is not None:
            return BAx_plus_Wx + self.b
        return BAx_plus_Wx

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


class LoRAHandle:
    def __init__(self, loras: Iterable[LoRA]) -> None:
        self.loras = loras

    def activate(self, name=None):
        for lora in self.loras:
            lora.activate(name)

    def deactivate(self):
        for lora in self.loras:
            lora.deactivate()

    def add_lora(self, name: str, active_init: bool = False):
        for lora in self.loras:
            lora.add_lora(name, active_init)

    def remove_lora(self, name: str):
        for lora in self.loras:
            lora.remove_lora(name)


def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: float = None,
    module_names: Iterable[str] = [],
    lora_names: Iterable[str] = [],
):
    """
    Apply LoRA to the model.
    The LoRA is applied to all linear layers in the model.

    Args:
        model (nn.Module):
            The model to apply LoRA.

        rank (int):
            The rank of the LoRA.

        alpha (int):
            The alpha value of the LoRA. (default: rank * 2)

        module_names (Iterable[str]):
            The names of the modules to apply LoRA.
            If empty, apply to all linear layers.

        lora_names (Iterable[str]):
            The names of the applied LoRAs.
            If empty, only one LoRA is applied.
    """
    _loras = []
    if isinstance(lora_names, str):
        lora_names = [
            lora_names,
        ]
    elif len(lora_names) == 0:
        lora_names = ["base"]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (
            len(module_names) == 0
            or any([module_name in name for module_name in module_names])
        ):
            _lora = LoRA(module, rank, alpha=alpha, name=lora_names)
            _loras.append(_lora)
            _name = name.split(".")
            _module = model
            for i in range(len(_name) - 1):
                _module = _module.__getattr__(_name[i])
            _module.__setattr__(_name[-1], _lora)
    return LoRAHandle(_loras)


def get_handle(model: nn.Module, module_names: Iterable[str] = []):
    """
    Get the LoRAHandle object from the model.
    This function is useful when you restore the model from the checkpoint.

    Args:
        model (nn.Module):
            The model to get the LoRAHandle.

        module_names (Iterable[str]):
            The names of the modules to get the LoRAHandle.
            If empty, get all LoRAHandle.
    """
    _loras = []
    for name, module in model.named_modules():
        if isinstance(module, LoRA) and (
            len(module_names) == 0
            or any([module_name in name for module_name in module_names])
        ):
            _loras.append(module)
    return LoRAHandle(_loras)
