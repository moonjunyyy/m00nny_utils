"""O-LoRA (Wang et al., EMNLP 2023): orthogonal subspace learning with LoRA.

A new adapter per task; the earlier ones are frozen and stay in the weight.
The new adapter is pushed orthogonal to every earlier one by

    aux_loss = lambda_orth * sum_layers sum_{i<t} sum_jk (M_i^T M_t)_jk ^ p
             + lambda_l2   * sum_layers (||A_t|| + ||B_t||),

so the update of task t lives in a subspace the earlier updates do not use.

orth_on picks the matrix M: 'down' is A (in, rank), the subspace of the
layer input the update reads from; 'up' is B (out, rank), the subspace of
the layer output it writes to.  orth_power is p: 2 sums squared entries
(the equation of the paper), 1 sums absolute entries.
"""
import torch
import torch.nn as nn

from ._task_lora import DEFAULT_TARGETS, _TaskLoRAHandler


class OLoRAHandler(_TaskLoRAHandler):
    def __init__(
        self,
        module: nn.Module,
        rank: int = 4,
        alpha: int = None,
        num_tasks: int = 10,
        module_names=DEFAULT_TARGETS,
        lambda_orth: float = 0.5,
        lambda_l2: float = 0.0,
        orth_on: str = 'down',
        orth_power: int = 2,
        **kwargs,
    ):
        super().__init__(
            module, rank=rank, alpha=alpha, num_tasks=num_tasks,
            module_names=module_names, **kwargs)
        assert orth_on in ('down', 'up')
        assert orth_power in (1, 2)
        self.lambda_orth = lambda_orth
        self.lambda_l2 = lambda_l2
        self.orth_on = orth_on
        self.orth_power = orth_power

    def _aux(self):
        loss = self._zero()
        _t = self.task_id
        for lora in self.layers:
            _side = lora._as if self.orth_on == 'down' else lora._bs
            if _t > 0 and self.lambda_orth:
                old = torch.cat(
                    [_side[str(_i)] for _i in range(_t)], dim=1).detach()
                overlap = old.T @ _side[str(_t)]
                overlap = overlap.abs() if self.orth_power == 1 \
                    else overlap.pow(2)
                loss = loss + self.lambda_orth * overlap.sum()
            if self.lambda_l2:
                loss = loss + self.lambda_l2 * (
                    lora._as[str(_t)].norm() + lora._bs[str(_t)].norm())
        return loss
