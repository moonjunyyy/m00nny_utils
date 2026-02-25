import math
import torch


class WarmUpCosineAnnelingScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lrs = None
        super(WarmUpCosineAnnelingScheduler, self).__init__(optimizer)

    def _initial_step(self):
        if self.initial_lrs is None:
            self.initial_lrs = []
            for group in self.optimizer.param_groups:
                self.initial_lrs.append(group["lr"])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self.initial_lrs[i]
        super(WarmUpCosineAnnelingScheduler, self)._initial_step()

    def get_lr(self):
        last_epoch = (self.last_epoch - self.warmup_steps) % (
            self.total_steps - self.warmup_steps
        )
        if self.last_epoch < self.warmup_steps:
            return [
                self.initial_lrs[i]
                * (self.last_epoch+1)
                / (self.warmup_steps+1)
                for i in range(len(self.optimizer.param_groups))
            ]
        else:
            return [
                0.5
                * (
                    1
                    + math.cos(
                        math.pi * last_epoch /
                        (self.total_steps - self.warmup_steps)
                    )
                )
                * self.initial_lrs[i]
                for i in range(len(self.optimizer.param_groups))
            ]
