import os
import time
import random
import inspect
import importlib
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

from ..system.log import Log
from ..parallel.load_parallel import (
    TensorParallelInfo,
    MultiprocessDeviceInfo
)


def get_dataset(
    name: str,
    path: str = "./data",
    download: bool = False,
    **kwargs
):
    pass


# Short names `name` may be given as, beyond the class names of torch.optim
# and torch.optim.lr_scheduler. A value is a class or its dotted import path;
# a project adds its own the same way.
_ROOT = __package__.rpartition(".")[0]
OPTIMIZERS = {}
SCHEDULERS = {
    "cosine": f"{_ROOT}.lr_scheduler.warmup_cosine_anneling"
              ".WarmUpCosineAnnelingScheduler",
    "constant": f"{_ROOT}.lr_scheduler.warmup_constant"
                ".WarmUpConstantScheduler",
}


def _resolve(name, table, namespace, base, suffix=""):
    """The class `name` stands for, or None.

    `name` is a class, a key of `table`, the name of a subclass of `base` in
    `namespace` (case ignored, `suffix` optional: "step" is StepLR), or a
    dotted import path.
    """
    if isinstance(name, type):
        return name
    _key = name.lower()
    _found = table.get(_key)
    if _found is None:
        _classes = {
            _attr.lower(): _obj for _attr, _obj in vars(namespace).items()
            if isinstance(_obj, type) and issubclass(_obj, base)
            and _obj is not base and not _attr.startswith("_")
        }
        _found = _classes.get(_key, _classes.get(_key + suffix))
    if _found is None and "." in name:
        _found = name
    if isinstance(_found, str):
        _module, _, _attr = _found.rpartition(".")
        _found = getattr(importlib.import_module(_module), _attr)
    return _found


def _accepted(target, options):
    """The entries of `options` that `target` takes by name, its first
    (positional) argument aside."""
    _names = list(inspect.signature(target).parameters)[1:]
    return {_k: _v for _k, _v in options.items() if _k in _names}


def get_optimizer(
        parameters,
        *,
        name: str = "adamw",
        lr: float = 1e-3,
        momentum: float = 0.8,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        betas: Tuple[float, float] = (0.9, 0.999),
        **kwargs
):
    _optimizer = _resolve(
        name, OPTIMIZERS, torch.optim, torch.optim.Optimizer
    )
    if _optimizer is None:
        raise ValueError(f"Unsupported optimizer: {name}")
    # each optimizer gets the options it takes: Adam has no momentum,
    # RMSprop no dampening
    return _optimizer(
        parameters,
        **_accepted(_optimizer, dict(
            kwargs,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            betas=betas,
        ))
    )


def get_scheduler(
    target_optimizer,
    *,
    name: str = "constant",
    warmup_steps: int = 0,
    total_steps: int = 1000,
    gamma: float = 0.9,
    step_size: int = 10,
    **kwargs
):
    _scheduler = _resolve(
        name, SCHEDULERS, torch.optim.lr_scheduler,
        torch.optim.lr_scheduler.LRScheduler, suffix="lr"
    )
    if _scheduler is None:
        raise ValueError(f"Unsupported scheduler: {name}")
    return _scheduler(
        target_optimizer,
        **_accepted(_scheduler, dict(
            kwargs,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            gamma=gamma,
            step_size=step_size,
        ))
    )


class RuntimeBase:
    def __init__(
        self,
        *,
        mode=None,
        epochs=100,
        batchsize=128,
        device="cuda",
        dtype=torch.float32,
        random_seed=None,
        num_workers=4,
        save_path="./checkpoints",
        dist={},
        opt={},
        lrs={},
        data={},
        **kwargs
    ):
        self.mode = mode
        self.epochs = epochs
        self.batchsize = batchsize
        self.device = device
        # the config passes the dtype by name ("float32", "bfloat16")
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.save_path = save_path

        self.dist_config = dist
        self.opt_config = opt
        self.lr_scheduler_config = lrs
        self.data_config = data
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        if self.random_seed is None:
            self.random_seed = time.time_ns() % 2 ** 32

    def init_distributed(self):
        _g_world = self.dist_config["global_world_size"]
        _g_rank = self.dist_config["global_rank"]
        _l_world = self.dist_config["local_world_size"]
        _l_rank = self.dist_config["local_rank"]
        _dev = torch.device(_l_rank if _l_world > 1 else "cpu")

        self.device_info = MultiprocessDeviceInfo(
            global_world_size=_g_world,
            local_world_size=_l_world,
            global_rank=_g_rank,
            local_rank=_l_rank,
        )
        time.sleep(0.1 * _l_rank)  # Stagger the initialization

        # the config names it "protocol"; "url" is the earlier key
        _dist_url = self.dist_config.get(
            "protocol", self.dist_config.get("url", "tcp://"))
        _master_addr = self.dist_config.get("master_addr", "localhost")
        _master_port = self.dist_config.get("master_port", "29500")
        _dist_url = f"{_dist_url}{_master_addr}:{_master_port}"
        _dist_bkend = self.dist_config.get("backend", "nccl")

        dist.init_process_group(
            backend=_dist_bkend,
            init_method=_dist_url,
            world_size=_g_world,
            rank=_g_rank,
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        import builtins as __builtin__
        # builtin_print = __builtin__.print
        is_master = self.global_rank == 0

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                self.log.info(
                    "\n".join(
                        map(str, args)
                    )
                )
        __builtin__.print = print

    def set_random_seed(self):
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    def run(self):
        _worker_fn = getattr(self, f"_{self.mode}_worker", self.worker)
        procs = []
        for idx in range(1, self.world_size):
            p = torch.multiprocessing.Process(
                target=_worker_fn, args=(idx,)
            )
            p.start()
            procs.append(p)
        self.worker(0)
        for p in procs:
            p.join()

    def worker(self, _idx):
        self.dist_config["global_rank"] += _idx
        self.dist_config["local_rank"] = _idx
        self.log = Log(
            name=f"worker_{self.local_rank}",
            path=self.args.save_dir,
            level=(
                Log.DEBUG if self.local_rank == 0 else Log.WARNING
            ),
            use_STDOUT=(self.local_rank == 0),
            use_STDERR=False,
        )
        self.init_distributed()
        self.set_random_seed()
