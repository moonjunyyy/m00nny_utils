import math
import torch

class WarmUpConstantScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.initial_lrs = None
        super().__init__(optimizer, last_epoch, verbose)

    def _initial_step(self):
        if self.initial_lrs is None: self.initial_lrs = [group['lr'] for group in self.optimizer.param_groups]
        else:
            for i, group in enumerate(self.optimizer.param_groups): group['lr'] = self.initial_lrs[i]
        super(WarmUpConstantScheduler, self)._initial_step()

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.initial_lrs[i] * self.last_epoch / self.warmup_steps for i in range(len(self.optimizer.param_groups))]
        else:
            return [self.initial_lrs[i] for i in range(len(self.optimizer.param_groups))]