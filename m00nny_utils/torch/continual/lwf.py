"""LwF (Li & Hoiem, 2016): learning without forgetting.

When a task after the first begins, the model as it stands is copied and
frozen.  While the task trains, the copy's outputs on the *new* inputs are
the targets of a distillation term on the old classes,

    aux_loss = lam * T^2 * KL( softmax(old / T) || softmax(new / T) ).

``aux_loss`` belongs to the latest forward through the handler.  The old
classes are the union of the `classes` given to ``before_task`` so far; when
none were ever given, the distillation runs over all outputs.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._cl_handler import _CLHandler


class LwFHandler(_CLHandler):
    def __init__(
        self,
        module: nn.Module,
        lam: float = 1.0,
        temperature: float = 2.0,
        **kwargs,
    ):
        super().__init__(module)
        self.lam = lam
        self.temperature = temperature
        self._teacher = []      # in a list: not a submodule, not a parameter
        self._seen = set()
        self._old_classes = None
        self._distill = None

    def before_task(self, task_id, classes=None, loader=None, **kwargs):
        self.task_id = task_id
        self._distill = None
        if task_id > 0:
            _teacher = copy.deepcopy(self.module).eval()
            for param in _teacher.parameters():
                param.requires_grad = False
            self._teacher = [_teacher]
            self._old_classes = sorted(self._seen) if self._seen else None
        if classes is not None:
            self._seen.update(int(_c) for _c in classes)

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        self._distill = None
        if self._enabled and self.training and self._teacher:
            with torch.no_grad():
                old = self._teacher[0](*args, **kwargs)
            new = out
            if self._old_classes is not None:
                old = old[:, self._old_classes]
                new = new[:, self._old_classes]
            _t = self.temperature
            self._distill = F.kl_div(
                F.log_softmax(new.float() / _t, dim=1),
                F.softmax(old.float() / _t, dim=1),
                reduction="batchmean",
            ) * (_t * _t)
        return out

    def _aux(self):
        if self._distill is None:
            return self._zero()
        return self.lam * self._distill
