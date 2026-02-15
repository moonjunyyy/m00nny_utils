import torch


class WarmUpConstantScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.initial_lrs = None
        super().__init__(optimizer, last_epoch)

    def _initial_step(self):
        if self.initial_lrs is None:
            for group in self.optimizer.param_groups:
                self.initial_lrs = [group["lr"]]
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self.initial_lrs[i]
        super(WarmUpConstantScheduler, self)._initial_step()

    def get_lr(self):
        ret = []
        if self.last_epoch < self.warmup_steps:
            for i in range(len(self.optimizer.param_groups)):
                ret.append(self.initial_lrs[i] *
                           self.last_epoch / self.warmup_steps)
        else:
            for i in range(len(self.optimizer.param_groups)):
                ret.append(self.initial_lrs[i])
        return ret
