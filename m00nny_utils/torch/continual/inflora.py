"""InfLoRA (Liang & Li, CVPR 2024): interference-free low-rank adaptation.

Orthogonality by construction instead of by a loss.  Before a task trains,
the down-projection A_t of every adapted weight is *designed* and then fixed:
its columns are the leading directions of the new task's layer inputs after
the directions the old tasks occupy have been projected out.  Only the
up-projection B_t (zero at the start) trains, so the update B_t A_t^T does
not move the layer's output on inputs lying in the old tasks' subspace.
After the task, that subspace (a GPM basis per layer) grows by the new
task's input directions up to an energy threshold.

Both steps need the input statistics of the adapted layers:

* ``before_task(..., loader=...)`` / ``after_task(..., loader=...)``: one
  forward pass over the loader each (the procedure of the paper).
* no loader: the statistics are gathered from the training forwards.  The
  first epoch of the task is then a warm-up with the adapter held at zero
  (only the head trains); A_t is designed at the first ``after_epoch``; the
  basis update uses the statistics of the last epoch.  Needs epochs >= 2.

The basis is kept as such; the complement representation of DualGPM, which
only saves memory once the basis passes half the dimension, is not kept.
"""
import math

import torch
import torch.nn as nn

from ._cl_handler import all_reduce_
from ._task_lora import DEFAULT_TARGETS, _LayerTaskLoRA, _TaskLoRAHandler


class _LayerInfLoRA(_LayerTaskLoRA):
    """Task-wise LoRA that keeps the input statistics of its own layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GPM basis of the layer's old-task input space, (in, k).
        self.register_buffer("basis", None)
        self.track = False      # gather from training forwards
        self.in_pass = False    # inside an explicit loader pass
        self._cov = None
        self._count = 0

    def _observe(self, x):
        if not (self.in_pass or (self.track and self.training)):
            return
        x = x.detach().reshape(-1, self.dim_in).float()
        self._cov = x.T @ x if self._cov is None else self._cov + x.T @ x
        self._count += x.shape[0]

    def reset(self):
        self._cov = None
        self._count = 0

    def covariance(self):
        """Mean input covariance (float64), summed over the ranks."""
        if self._count == 0:
            raise RuntimeError(
                "InfLoRA has no input statistics: pass loader=..., or run at "
                "least one training epoch through the handler first")
        _count = all_reduce_(torch.tensor(
            float(self._count), device=self._cov.device))
        return all_reduce_(self._cov.clone()).double() / _count.double()

    def _project_out(self, cov):
        if self.basis is None:
            return cov
        _basis = self.basis.double()
        cov = cov - _basis @ (_basis.T @ cov)
        return cov - (cov @ _basis) @ _basis.T

    @torch.no_grad()
    def design(self):
        """Fix A of the current task from the gathered statistics."""
        _name = str(self.task_id)
        _, vecs = torch.linalg.eigh(self._project_out(self.covariance()))
        down = vecs[:, -self.rank:].flip(1)
        if self.basis is not None:
            _basis = self.basis.double()
            down = down - _basis @ (_basis.T @ down)
        self._as[_name].copy_((down / math.sqrt(3)).to(self._as[_name]))
        self._bs[_name].zero_()
        self.set_trainable(_name, down=False, up=True)

    @torch.no_grad()
    def update_basis(self, eps: float):
        """Grow the basis until it holds `eps` of the gathered input energy."""
        cov = self.covariance()
        _dim = cov.shape[0]
        _total = torch.trace(cov)
        rest = self._project_out(cov)
        _kept = _total - torch.trace(rest)
        if _kept >= eps * _total:
            return
        vals, vecs = torch.linalg.eigh(rest)
        vals, vecs = vals.flip(0).clamp_min(0), vecs.flip(1)
        _k = int((_kept + vals.cumsum(0) < eps * _total).sum()) + 1
        _k = min(_k, _dim - (0 if self.basis is None else self.basis.shape[1]))
        if _k <= 0:
            return
        new = vecs[:, :_k]
        if self.basis is not None:
            new, _ = torch.linalg.qr(torch.cat((self.basis.double(), new), 1))
        self.basis = new.to(self.weight.dtype).contiguous()


class InfLoRAHandler(_TaskLoRAHandler):
    def __init__(
        self,
        module: nn.Module,
        rank: int = 10,
        alpha: int = None,
        num_tasks: int = 10,
        module_names=DEFAULT_TARGETS,
        eps: float = 0.95,
        eps_end: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            module, rank=rank, alpha=alpha, num_tasks=num_tasks,
            module_names=module_names, **kwargs)
        # Energy threshold of the basis update: eps at the first task,
        # rising linearly toward eps_end.
        self.eps = eps
        self.eps_end = eps_end
        self._warm = False      # adapter of the current task not designed yet

    def _layer(self, module, weight_name, row_mask):
        return _LayerInfLoRA(
            module, weight_name, rank=self.rank, alpha=self.alpha,
            num_tasks=self.num_tasks, row_mask=row_mask)

    def _reset(self):
        for layer in self.layers:
            layer.reset()

    @torch.no_grad()
    def _pass(self, loader):
        _was_training = self.module.training
        self.module.eval()
        self._reset()
        _weight = self.layers[0].weight
        for layer in self.layers:
            layer.in_pass = True
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            self.module(x.to(device=_weight.device, dtype=_weight.dtype))
        for layer in self.layers:
            layer.in_pass = False
        self.module.train(_was_training)

    def _design(self):
        for layer in self.layers:
            layer.design()
        self._warm = False

    # ── the calls a training loop makes ─────────────────────────────────
    def before_task(self, task_id, classes=None, loader=None, **kwargs):
        self.task_id = task_id
        for layer in self.layers:
            layer.track = loader is None
        if loader is not None:
            self._pass(loader)
            self._design()
        else:
            self._warm = True
            self._reset()
            for layer in self.layers:
                layer.set_trainable(str(task_id), down=False, up=False)

    def before_epoch(self, epoch, **kwargs):
        if self.layers[0].track:
            self._reset()

    def after_epoch(self, epoch, **kwargs):
        if self._warm:
            self._design()

    def after_task(self, task_id, classes=None, loader=None, **kwargs):
        if self._warm:
            raise RuntimeError(
                "InfLoRA: the task ended before its adapter was designed "
                "(no loader in before_task and no after_epoch call)")
        if loader is not None:
            self._pass(loader)
        _eps = self.eps + (self.eps_end - self.eps) * task_id / self.num_tasks
        for layer in self.layers:
            layer.update_basis(_eps)
        self._reset()
