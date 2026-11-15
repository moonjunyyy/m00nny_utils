"""LoRA with one adapter per task, shared by the O-LoRA, InfLoRA and SD-LoRA
handlers.

The layer handler is ``lora._LayerLoRA`` with the adapters named after the
tasks ("0", "1", ...).  Every adapter up to the current task is in the
weight,

    W + scale * sum_{i <= t} B_i A_i^T,

because a class-incremental model has no task identity at test time; only
the adapter of the current task trains.
"""
import math
from typing import Iterable

import torch
import torch.nn as nn

from ._cl_handler import _CLHandler
from ..layer.lora import DEFAULT_TARGETS, _LayerLoRA, lora_targets


class _LayerTaskLoRA(_LayerLoRA):
    def __init__(
        self,
        module: nn.Module,
        weight_name: str = None,
        rank: int = 4,
        alpha: int = None,
        num_tasks: int = 1,
        row_mask: torch.Tensor = None,
    ):
        super().__init__(
            module, weight_name, rank=rank, alpha=alpha,
            names=[str(_t) for _t in range(num_tasks)], row_mask=row_mask)
        self._task_id = 0
        self.set_task(0)

    def _init_adapter(self):
        _like = dict(device=self.weight.device, dtype=self.weight.dtype)
        _bound = 1.0 / math.sqrt(self.dim_in)
        return (torch.empty(self.dim_in, self.rank, **_like).uniform_(
                    -_bound, _bound),
                torch.zeros(self.dim_out, self.rank, **_like))

    @property
    def task_id(self):
        return self._task_id

    def set_task(self, task_id: int):
        _name = str(task_id)
        if _name not in self._as:
            # Past num_tasks: the adapter is made now, so an optimizer built
            # before this call does not hold it.
            self.add_lora(_name)
        self._task_id = task_id
        _names = self.names()
        self.active = _names[:_names.index(_name) + 1]
        self.set_trainable(_name)


class _TaskLoRAHandler(_CLHandler):
    """A task-wise LoRA layer handler on every weight `module_names` selects.

    The adapters of all `num_tasks` tasks exist from the start.  With
    qv_only, a fused q/k/v projection is adapted in its query and value rows
    only.  The backbone is frozen (heads excepted) unless
    freeze_backbone=False; the adapted weights are frozen either way.
    """

    def __init__(
        self,
        module: nn.Module,
        rank: int = 4,
        alpha: int = None,
        num_tasks: int = 10,
        module_names: Iterable[str] = DEFAULT_TARGETS,
        qv_only: bool = True,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        super().__init__(module)
        if freeze_backbone:
            self.freeze_backbone()
        self.rank = rank
        self.alpha = alpha
        self.num_tasks = num_tasks
        self._prepare(**kwargs)
        _targets = lora_targets(module, module_names, qv_only)
        if not _targets:
            raise ValueError(
                f"nothing in {type(module).__name__} matches "
                f"module_names={list(module_names)}")
        for _, _sub, _weight_name, _mask in _targets:
            self.layers.append(self._layer(_sub, _weight_name, _mask))
        self.task_id = 0

    def _prepare(self, **kwargs):
        """Called before the layer handlers are made, for what _layer needs."""

    def _layer(self, module: nn.Module, weight_name: str, row_mask):
        """The layer handler for one selected weight."""
        return _LayerTaskLoRA(
            module, weight_name, rank=self.rank, alpha=self.alpha,
            num_tasks=self.num_tasks, row_mask=row_mask)

    @property
    def loras(self):
        return self.layers
