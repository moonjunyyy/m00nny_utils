"""Common ground for the continual-learning handlers of this folder.

Every handler wraps a module the way ``RainbowPromptHandler`` does, and
answers the same calls from a training loop::

    model = SomeHandler(backbone, ...)
    for task_id in range(num_tasks):
        model.before_task(task_id, classes=classes_of_task, loader=loader)
        for epoch in range(epochs):
            model.before_epoch(epoch)
            for x, y in loader:
                loss = criterion(model(x), y) + model.aux_loss
                ...
            model.after_epoch(epoch)
        model.after_task(task_id, classes=classes_of_task, loader=loader)

``classes`` and ``loader`` are optional keywords: a handler that can use them
says so in its own docstring, and every handler accepts and ignores the ones
it has no use for.  ``aux_loss`` is the method's own loss term (what
``RainbowPromptHandler`` calls ``similarity``).

The shape every handler has:

* It never replaces a part of the wrapped module.  What it adds lives in the
  handler; where it must intervene, a layer handler puts its own function in
  place of one module's ``forward`` and keeps the original (as
  ``_LayerPromptTuning`` does).  The layer handlers are in ``self.layers``.
* ``disable()`` takes every intervention out: the wrapped module then
  computes exactly what it computed before it was wrapped, and ``aux_loss``
  is zero.  ``enable()`` puts them back.
* Parameters a method needs for later tasks are allocated when the handler
  is built, never inside ``before_task``: an optimizer made from
  ``model.parameters()`` before ``before_task`` already holds them.  Which
  of them train during a task is decided by ``requires_grad`` alone.
"""
import re

import torch
import torch.distributed as dist
import torch.nn as nn

from ._hook_handler import _MetaPEFTHandler

# Parameter names that belong to an output head rather than to the backbone.
HEAD_PATTERN = r"^(head|heads|fc|classifier|predictor)\."


class _CLHandler(_MetaPEFTHandler):
    def __init__(self, module: nn.Module, *args, **kwargs):
        super().__init__(module)
        self._task_id = 0
        self.layers = nn.ModuleList()

    # ── in / out ────────────────────────────────────────────────────────
    def enable(self):
        self._enabled = True
        for layer in self.layers:
            layer.enable()

    def disable(self):
        self._enabled = False
        for layer in self.layers:
            layer.disable()

    # ── the calls a training loop makes ─────────────────────────────────
    def before_task(self, task_id, classes=None, loader=None, **kwargs):
        self.task_id = task_id

    def after_task(self, task_id, classes=None, loader=None, **kwargs):
        pass

    def before_epoch(self, epoch, **kwargs):
        pass

    def after_epoch(self, epoch, **kwargs):
        pass

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        self._task_id = value
        self._set_task(value)

    def _set_task(self, task_id):
        for layer in self.layers:
            layer.set_task(task_id)

    @property
    def aux_loss(self):
        return self._aux() if self._enabled else self._zero()

    def _aux(self):
        """The method's own loss term; a handler with one overrides this."""
        return self._zero()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # ── helpers ─────────────────────────────────────────────────────────
    def _zero(self):
        for p in self.module.parameters():
            return p.new_zeros(())
        return torch.zeros(())

    def freeze_backbone(self, head_pattern: str = HEAD_PATTERN):
        """Stop the gradient of every wrapped parameter that is not a head."""
        for name, param in self.module.named_parameters():
            if not re.match(head_pattern, name):
                param.requires_grad = False

    def __getattr__(self, name):
        # What the handler does not define is looked up on the wrapped module
        # (classifier, forward_features, num_classes, ...).
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = super().__getattr__("module")
            return getattr(module, name)


def all_reduce_(tensor: torch.Tensor) -> torch.Tensor:
    """Sum `tensor` over the ranks in place; nothing happens on one process."""
    if dist.is_available() and dist.is_initialized() \
            and dist.get_world_size() > 1:
        dist.all_reduce(tensor)
    return tensor


# ── ViT access: torchvision (encoder.layers) and timm (blocks) ──────────
def vit_layers(module: nn.Module):
    if hasattr(module, "blocks"):
        return list(module.blocks)
    if hasattr(module, "encoder") and hasattr(module.encoder, "layers"):
        return list(module.encoder.layers)
    raise TypeError(
        f"{type(module).__name__} has neither .blocks (timm) nor "
        ".encoder.layers (torchvision)")


def vit_attention(layer: nn.Module):
    for name, sub in layer.named_modules():
        if re.match(r".*[Aa]tte*n.*", name):
            return sub
    raise TypeError(f"no attention module in {type(layer).__name__}")


def vit_dims(module: nn.Module):
    """(embed_dim, num_heads) read from the first attention module."""
    attn = vit_attention(vit_layers(module)[0])
    embed_dim = getattr(attn, "embed_dim", None)
    if embed_dim is None:
        embed_dim = attn.qkv.in_features
    return embed_dim, attn.num_heads


def vit_tokens(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Normalized output tokens (B, N, C), class token first."""
    if hasattr(module, "forward_features"):
        return module.forward_features(x)
    x = module._process_input(x)
    cls = module.class_token.expand(x.shape[0], -1, -1)
    return module.encoder(torch.cat((cls, x), dim=1))


def vit_head(module: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    if hasattr(module, "forward_head"):
        return module.forward_head(tokens)
    return module.heads(tokens[:, 0])
