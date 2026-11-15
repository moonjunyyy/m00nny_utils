"""SD-LoRA (Wu et al., ICLR 2025): scalable decoupled low-rank adaptation.

Each task adds one LoRA component, and a component is split into a
direction (the update normalised to unit Frobenius norm) and a scalar
magnitude:

    W = W0 + sum_{k <= t} alpha_k * (B_k A_k^T) / ||B_k A_k^T||_F .

While task t trains, the directions of the earlier tasks are fixed, the new
direction is learned, and *all* magnitudes alpha_1 .. alpha_t are learned.
No rehearsal, no component selection at test time, no auxiliary loss.

Following the paper: magnitudes start at one; with a fused q/k/v projection
the query and the value part each get their own direction and share the
magnitude.  Because the direction is normalised, both A and B start random
(a zero B would make it undefined).  share_magnitudes=True keeps one alpha_k
for the whole network; False keeps one per adapted weight.

The rank-reduction (RR) and knowledge-distillation (KD) variants of the
paper, which only save parameters, are not here.
"""
import math

import torch
import torch.nn as nn

from ._task_lora import DEFAULT_TARGETS, _LayerTaskLoRA, _TaskLoRAHandler


class _LayerSDLoRA(_LayerTaskLoRA):
    def __init__(
        self,
        module: nn.Module,
        weight_name: str = None,
        rank: int = 10,
        num_tasks: int = 1,
        row_mask: torch.Tensor = None,
        magnitudes: nn.ParameterList = None,
    ):
        _weight = module._parameters[weight_name]
        _out, _in = _weight.shape
        # Row blocks of the weight; each adapted block is its own direction.
        if _out == 3 * _in:
            _blocks = [(_i * _in, (_i + 1) * _in) for _i in range(3)]
        else:
            _blocks = [(0, _out)]
        _adapted = [row_mask is None or bool(row_mask[_s] > 0)
                    for _s, _ in _blocks]
        _groups = sum(_adapted)
        # A shared list belongs to the handler and is only referred to here.
        self.__dict__["_shared"] = () if magnitudes is None else (magnitudes,)
        super().__init__(module, weight_name, rank=rank * _groups, alpha=None,
                         num_tasks=num_tasks, row_mask=row_mask)
        self.blocks = _blocks
        # Columns of B a row block may use: its own `rank` of them.
        _cols = torch.zeros(_out, rank * _groups,
                            device=_weight.device, dtype=_weight.dtype)
        _g = 0
        for (_s, _e), _on in zip(_blocks, _adapted):
            if _on:
                _cols[_s:_e, _g * rank:(_g + 1) * rank] = 1.0
                _g += 1
        self.register_buffer("col_mask", _cols)
        if magnitudes is None:
            self.magnitudes = nn.ParameterList([
                nn.Parameter(torch.ones((), device=_weight.device,
                                        dtype=_weight.dtype))
                for _ in range(num_tasks)])
            self.set_task(self.task_id)

    def _init_adapter(self):
        _like = dict(device=self.weight.device, dtype=self.weight.dtype)
        _a, _b = 1.0 / math.sqrt(self.dim_in), 1.0 / math.sqrt(self.rank)
        return (torch.empty(self.dim_in, self.rank, **_like).uniform_(-_a, _a),
                torch.empty(self.dim_out, self.rank, **_like).uniform_(-_b, _b))

    def set_task(self, task_id: int):
        super().set_task(task_id)
        if not self._shared and "magnitudes" in self._modules:
            for _i, _m in enumerate(self.magnitudes):
                _m.requires_grad = _i <= task_id

    def magnitude(self, name: str):
        _list = self._shared[0] if self._shared else self.magnitudes
        return _list[int(name)]

    def direction(self, name: str):
        """B A^T of one task, each row block scaled to unit Frobenius norm."""
        _update = (self._bs[name] * self.col_mask) @ self._as[name].T
        return torch.cat([
            _update[_s:_e] / _update[_s:_e].norm().clamp_min(1e-12)
            for _s, _e in self.blocks
        ], dim=0)

    def delta(self):
        _delta = 0
        for _n in self.active:
            _delta = _delta + self.magnitude(_n) * self.direction(_n)
        return _delta


class SDLoRAHandler(_TaskLoRAHandler):
    def __init__(
        self,
        module: nn.Module,
        rank: int = 10,
        num_tasks: int = 10,
        module_names=DEFAULT_TARGETS,
        share_magnitudes: bool = True,
        **kwargs,
    ):
        super().__init__(
            module, rank=rank, num_tasks=num_tasks,
            module_names=module_names,
            share_magnitudes=share_magnitudes, **kwargs)

    def _prepare(self, share_magnitudes: bool = True, **kwargs):
        self.magnitudes = None
        if share_magnitudes:
            _ref = self._zero()
            self.magnitudes = nn.ParameterList([
                nn.Parameter(torch.ones((), device=_ref.device,
                                        dtype=_ref.dtype))
                for _ in range(self.num_tasks)])

    def _layer(self, module, weight_name, row_mask):
        return _LayerSDLoRA(
            module, weight_name, rank=self.rank, num_tasks=self.num_tasks,
            row_mask=row_mask, magnitudes=self.magnitudes)

    def _set_task(self, task_id: int):
        if task_id >= self.num_tasks:
            raise ValueError(
                f"SD-LoRA was built for {self.num_tasks} tasks; "
                f"task {task_id} has no magnitude")
        super()._set_task(task_id)
        if self.magnitudes is not None:
            for _i, _m in enumerate(self.magnitudes):
                _m.requires_grad = _i <= task_id
