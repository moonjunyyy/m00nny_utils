"""LoRA.

Handler form — ``_LayerLoRA`` and ``LoRAHandler``.  A layer handler attaches
to one weight matrix the way ``_LayerPromptTuning`` attaches to one attention
module: the module stays where it is, with its own weight; the handler owns
the low-rank factors and, while enabled, puts its own function in place of
the module's ``forward``.  That function lets the module see

    W + scale * sum_{n in active} B_n A_n^T

for the length of the call and then runs the original forward.  Because the
weight is reached by name, any matrix a module reads as an attribute can be
adapted: ``weight`` of an ``nn.Linear``, ``in_proj_weight`` of an
``nn.MultiheadAttention``.

Module-replacing form — ``LoRA``, ``LoRAHandle``, ``apply_lora``,
``get_handle``, further down — is the earlier one, kept for code that calls
it: it swaps each ``nn.Linear`` for a ``LoRA`` module.
"""
import re
from typing import Iterable, Union

import torch
import torch.nn as nn

from ._hook_handler import _MetaPEFTHandler

# The attention input projections of a ViT: timm (a fused nn.Linear) and
# torchvision (the in_proj_weight of an nn.MultiheadAttention).
DEFAULT_TARGETS = (r".*attn\.qkv$", r".*self_attention$")


class _LayerLoRA(_MetaPEFTHandler):
    """LoRA on one weight matrix of `module`.

    A_n = ``_as[n]`` (in, rank) is the down-projection and B_n = ``_bs[n]``
    (out, rank) the up-projection of the adapter named n; ``active`` lists
    the adapters that are in the weight.  The adapted weight itself (and its
    bias) no longer trains.
    """

    def __init__(
        self,
        module: nn.Module,
        weight_name: str = None,
        rank: int = 4,
        alpha: int = None,
        names: Iterable[str] = ("base",),
        row_mask: torch.Tensor = None,
    ):
        super().__init__(module)
        if weight_name is None:
            weight_name = "weight" if isinstance(module, nn.Linear) \
                else "in_proj_weight"
        self.weight_name = weight_name
        self.weight.requires_grad = False
        _bias = module._parameters.get(weight_name.replace("weight", "bias"))
        if _bias is not None:
            _bias.requires_grad = False
        self.dim_out, self.dim_in = self.weight.shape
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.lora_scale = self.alpha / self.rank
        # (out, 1) of 0/1: rows of the weight the adapters may change.
        self.register_buffer("row_mask", row_mask)
        self._as = nn.ParameterDict()
        self._bs = nn.ParameterDict()
        names = (names,) if isinstance(names, str) else tuple(names)
        for _n in names:
            self.add_lora(_n)
        self.active = [names[0]]
        self._original_forward = self.module.forward
        self.enable()

    @property
    def weight(self):
        return self.module._parameters[self.weight_name]

    # ── in / out ────────────────────────────────────────────────────────
    def enable(self):
        self._enabled = True
        self.module.forward = self._adapted_forward

    def disable(self):
        self._enabled = False
        self.module.forward = self._original_forward

    def _adapted_forward(self, *args, **kwargs):
        if args:
            self._observe(args[0])
        _weight = self.weight
        # An instance attribute shadows the registered parameter of the same
        # name, so the module's own code reads the adapted matrix.
        self.module.__dict__[self.weight_name] = \
            _weight + self.delta().to(_weight.dtype)
        try:
            return self._original_forward(*args, **kwargs)
        finally:
            del self.module.__dict__[self.weight_name]

    def _observe(self, x):
        """Sees the module's input on every adapted call; unused here."""

    # ── adapters ────────────────────────────────────────────────────────
    def _init_adapter(self):
        _like = dict(device=self.weight.device, dtype=self.weight.dtype)
        return (torch.randn(self.dim_in, self.rank, **_like),
                torch.zeros(self.dim_out, self.rank, **_like))

    def add_lora(self, name: str):
        _a, _b = self._init_adapter()
        self._as[name] = nn.Parameter(_a)
        self._bs[name] = nn.Parameter(_b)

    def remove_lora(self, name: str):
        if name in self._as:
            del self._as[name]
            del self._bs[name]
        self.active = [_n for _n in self.active if _n != name]

    def activate(self, name: str = None):
        self.active = [name if name is not None else next(iter(self._as))]

    def names(self):
        return list(self._as.keys())

    def set_trainable(self, name: str, down: bool = True, up: bool = True):
        """Only adapter `name` trains; `down` / `up` choose which half."""
        for _n in self._as.keys():
            self._as[_n].requires_grad = down and _n == name
            self._bs[_n].requires_grad = up and _n == name

    def delta(self):
        if not self.active:
            return self.weight.new_zeros(())
        _a = torch.cat([self._as[_n] for _n in self.active], dim=1)
        _b = torch.cat([self._bs[_n] for _n in self.active], dim=1)
        _delta = (_b @ _a.T) * self.lora_scale
        if self.row_mask is not None:
            _delta = _delta * self.row_mask
        return _delta


def lora_targets(module: nn.Module, module_names: Iterable[str],
                 qv_only: bool = False):
    """The weights the names select, as (name, module, weight_name, row_mask).

    A name matches as in ``re.match`` (an empty list matches everything).  A
    matching ``nn.Linear`` gives its ``weight``, a matching
    ``nn.MultiheadAttention`` its ``in_proj_weight``.  With qv_only, a fused
    q/k/v projection (out = 3 * in) gets a row mask that leaves the key rows
    alone.
    """
    _pattern = "|".join(module_names)
    _attentions = [name for name, sub in module.named_modules()
                   if isinstance(sub, nn.MultiheadAttention)]
    targets = []
    for name, sub in module.named_modules():
        if not re.match(_pattern, name):
            continue
        if isinstance(sub, nn.MultiheadAttention):
            if sub.in_proj_weight is None:
                continue
            _weight_name = "in_proj_weight"
        elif isinstance(sub, nn.Linear):
            # out_proj of an nn.MultiheadAttention is read as a matrix by the
            # attention, never called: its forward is not a place to stand.
            if any(name.startswith(_a + ".") for _a in _attentions):
                continue
            _weight_name = "weight"
        else:
            continue
        _weight = getattr(sub, _weight_name)
        _mask = None
        if qv_only and _weight.shape[0] == 3 * _weight.shape[1]:
            _mask = torch.ones(_weight.shape[0], 1, device=_weight.device,
                               dtype=_weight.dtype)
            _mask[_weight.shape[1]:2 * _weight.shape[1]] = 0.0
        targets.append((name, sub, _weight_name, _mask))
    return targets


class LoRAHandler(_MetaPEFTHandler):
    """LoRA on every weight `module_names` selects (see lora_targets).

    One adapter is active at a time, the first of `lora_names` to begin
    with; ``activate(name)`` changes it, ``disable()`` gives the bare model.
    """

    def __init__(
        self, module, rank, alpha=None,
        module_names=[], lora_names=[], **kwargs
    ):
        super().__init__(module, **kwargs)
        if isinstance(lora_names, str):
            lora_names = [lora_names]
        elif len(lora_names) == 0:
            lora_names = ["base"]
        self.loras = nn.ModuleList([
            _LayerLoRA(_sub, _weight_name, rank=rank, alpha=alpha,
                       names=lora_names)
            for _, _sub, _weight_name, _ in lora_targets(module, module_names)
        ])

    def enable(self):
        self._enabled = True
        for lora in self.loras:
            lora.enable()

    def disable(self):
        self._enabled = False
        for lora in self.loras:
            lora.disable()

    def activate(self, name=None):
        for lora in self.loras:
            lora.activate(name)

    def add_lora(self, name: str):
        for lora in self.loras:
            lora.add_lora(name)

    def remove_lora(self, name: str):
        for lora in self.loras:
            lora.remove_lora(name)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


# ── module-replacing form (the earlier one, kept for its callers) ───────


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
