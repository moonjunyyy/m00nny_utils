import os
import time
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.distributed as dist

from ..system.log import Log


def get_model(
    model: str,
    pretrained: str = None,
    **kwargs
):
    import torchvision.models as models
    if pretrained is not None:
        model_weights = getattr(models, f"{model}_Weights")
        model_weights = getattr(model_weights, pretrained.upper())
        model = getattr(models, model.lower())(weights=model_weights)
    else:
        model = getattr(models, model.lower())()
    return model


def get_dataset(
    dataset,
    download: bool = False,
    train_transform=None,
    test_transform=None,
    target_transform=None,
    **kwargs
):
    from ..dataset.color import get_rgb_dataset
    from ..dataset.gray import get_grayscale_dataset
    trainset = get_rgb_dataset(
        name=dataset,
        download=download,
        train=True,
        transform=train_transform,
        target_transform=target_transform,
    )
    testset = get_rgb_dataset(
        name=dataset,
        download=download,
        train=False,
        transform=test_transform,
        target_transform=target_transform,
    )
    num_classes = len(trainset.classes)
    return trainset, testset, num_classes


def get_optimizer(
        parameters,
        *,
        optimizer: str,
        lr: float = 1e-3,
        momentum: float = 0.8,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        betas: Tuple[float, float] = (0.9, 0.999),
        **kwargs
):
    if optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    elif optimizer.lower() == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    elif optimizer.lower() == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
    elif optimizer.lower() == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")


def get_scheduler(
    target_optimizer,
    *,
    lr_scheduler: str = "constant",
    warmup_steps: int = 0,
    total_steps: int = 1000,
    alpha: float = 0.9,
    gamma: float = 0.9,
    step_size: int = 10,
    **kwargs
):
    if lr_scheduler.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            target_optimizer, step_size=step_size, gamma=gamma
        )
    elif lr_scheduler.lower() == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR
    elif lr_scheduler.lower() == "cosine":
        from ..lr_scheduler.warmup_cosine_anneling \
            import WarmUpCosineAnnelingScheduler
        return WarmUpCosineAnnelingScheduler(
            target_optimizer, warmup_steps=warmup_steps, total_steps=total_steps
        )
        return WarmUpCosineAnnelingScheduler
    elif lr_scheduler.lower() == "constant":
        from ..lr_scheduler.warmup_constant import WarmUpConstantScheduler
        return WarmUpConstantScheduler(
            target_optimizer, warmup_steps=warmup_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler: {lr_scheduler}")


class _MetaTrainer:
    def __init__(self, args):
        self.args = args
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir, exist_ok=True)

        self.world_size = self.args.world_size
        self.global_rank = self.args.global_rank
        self.local_rank = -1
        self.dist_backend = self.args.dist_backend
        self.dist_url = (
            self.args.dist_url +
            self.args.dist_master_addr +
            ":" + self.args.dist_master_port
        )
        self.device = torch.device(self.args.device)
        self.dtype = getattr(torch, self.args.dtype)
        self.set_random_seed()
        print(self.args)

    def init_distributed(self):
        time.sleep(0.1 * self.local_rank)
        dist.init_process_group(
            backend=self.dist_backend,
            init_method=self.dist_url,
            world_size=self.world_size,
            rank=self.global_rank,
        )
        torch.cuda.set_device(self.local_rank)

        import builtins as __builtin__
        builtin_print = __builtin__.print

        is_master = self.global_rank == 0

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

    def set_random_seed(self):
        if self.args.random_seed is not None:
            self.args.random_seed = time.time_ns() % 2 ** 32
        torch.manual_seed(self.args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.args.random_seed)
        random.seed(self.args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.random_seed)

    def worker(self, _idx):
        self.local_rank = _idx
        self.global_rank += _idx
        self.init_distributed()
        self.set_random_seed()
        self.log = Log(
            name=f"worker_{self.local_rank}",
            path=self.args.save_dir,
            level=(
                Log.DEBUG if self.local_rank == 0 else Log.WARNING
            ),
            use_STDOUT=(self.local_rank == 0),
            use_STDERR=False,
        )
        self.device = torch.device(f"cuda:{self.local_rank}")
