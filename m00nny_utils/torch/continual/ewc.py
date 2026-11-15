"""EWC (Kirkpatrick et al., 2017), in its online form (Schwarz et al., 2018).

After each task the parameters are anchored and their diagonal Fisher
information is folded into one running importance,

    F <- gamma * F + F_task,        aux_loss = lam / 2 * sum F (theta - anchor)^2.

The Fisher of a task comes from one of two places:

* ``after_task(..., loader=...)``: a pass over the loader, squared gradient
  of the cross-entropy per batch (restricted to `classes` when given).
* no loader: the running mean of the squared gradients seen while the task
  trained, last epoch only (the EWC++ estimate).  Every backward through the
  parameters is counted, the penalty's own gradient included.

Only the parameters that require a gradient when the handler is built are
tracked.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from ._cl_handler import _CLHandler, all_reduce_


class EWCHandler(_CLHandler):
    def __init__(
        self,
        module: nn.Module,
        lam: float = 5000.0,
        gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(module)
        self.lam = lam
        self.gamma = gamma
        self._tracked = {
            name: param for name, param in module.named_parameters()
            if param.requires_grad
        }
        self._keys = {name: name.replace(".", "__") for name in self._tracked}
        self._online = {}
        self._steps = 0
        self._tracking = True
        self._first = next(iter(self._tracked))
        for name in self._tracked:
            self.register_buffer(f"fisher__{self._keys[name]}", None)
            self.register_buffer(f"anchor__{self._keys[name]}", None)
        self._hooks = []
        self.enable()

    def enable(self):
        self._enabled = True
        if not self._hooks:
            self._hooks = [
                param.register_hook(partial(self._on_grad, name))
                for name, param in self._tracked.items()
            ]

    def disable(self):
        self._enabled = False
        for _hook in self._hooks:
            _hook.remove()
        self._hooks = []

    def _fisher(self, name):
        return getattr(self, f"fisher__{self._keys[name]}")

    def _anchor(self, name):
        return getattr(self, f"anchor__{self._keys[name]}")

    # ── Fisher from the training gradients ──────────────────────────────
    def _on_grad(self, name, grad):
        if self._tracking and self.module.training:
            _sq = grad.detach().float().pow(2)
            if name in self._online:
                self._online[name] += _sq
            else:
                self._online[name] = _sq
            if name == self._first:
                self._steps += 1
        return grad

    def _reset(self):
        self._online = {}
        self._steps = 0

    # ── Fisher from a pass over a loader ────────────────────────────────
    def _fisher_pass(self, loader, classes):
        _was_training = self.module.training
        self.module.eval()
        # Gradients are taken at detached copies, so the hooks on the real
        # parameters (this one, a distributed gradient hook) stay silent.
        _copies = {name: param.detach().clone().requires_grad_(True)
                   for name, param in self._tracked.items()}
        _ref = self._tracked[self._first]
        _fisher = {name: torch.zeros_like(param, dtype=torch.float32)
                   for name, param in self._tracked.items()}
        _steps = 0
        for x, y in loader:
            x = x.to(device=_ref.device, dtype=_ref.dtype)
            y = y.to(device=_ref.device)
            logits = functional_call(self.module, _copies, (x,)).float()
            if classes is not None:
                _mask = torch.full_like(logits[:1], float("-inf"))
                _mask[:, list(classes)] = 0.0
                logits = logits + _mask
            grads = torch.autograd.grad(
                F.cross_entropy(logits, y), list(_copies.values()),
                allow_unused=True)
            for name, grad in zip(_copies, grads):
                if grad is not None:
                    _fisher[name] += grad.float().pow(2)
            _steps += 1
        self.module.train(_was_training)
        return _fisher, _steps

    # ── the calls a training loop makes ─────────────────────────────────
    def before_epoch(self, epoch, **kwargs):
        self._reset()

    @torch.no_grad()
    def _consolidate(self, fisher, steps):
        if steps == 0:
            raise RuntimeError(
                "EWC has no gradients to build the Fisher from: pass "
                "loader=..., or train through the handler first")
        _steps = None
        for name, param in self._tracked.items():
            _new = fisher.get(name)
            if _new is None:
                _new = torch.zeros_like(param, dtype=torch.float32)
            if _steps is None:
                _steps = all_reduce_(torch.tensor(
                    float(steps), device=_new.device))
            _new = all_reduce_(_new.clone()) / _steps
            _old = self._fisher(name)
            if _old is not None:
                _new = self.gamma * _old + _new
            setattr(self, f"fisher__{self._keys[name]}", _new)
            setattr(self, f"anchor__{self._keys[name]}",
                    param.detach().clone())

    def after_task(self, task_id, classes=None, loader=None, **kwargs):
        if loader is not None:
            self._tracking = False
            fisher, steps = self._fisher_pass(loader, classes)
            self._tracking = True
        else:
            fisher, steps = self._online, self._steps
        self._consolidate(fisher, steps)
        self._reset()

    def _aux(self):
        loss = self._zero()
        for name, param in self._tracked.items():
            _fisher = self._fisher(name)
            if _fisher is not None:
                loss = loss + (
                    _fisher * (param - self._anchor(name)).float().pow(2)
                ).sum()
        return 0.5 * self.lam * loss
