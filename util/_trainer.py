import time
import random

import numpy as np
import torch
import torch.distributed as dist

from ..torch.parallel.sharded_modules import load_transformers_as_sharded_module
from ..system.log import Log
from dataclasses import dataclass


@dataclass
class _MetaTrainerConfig:
    model_name: str
    optimizer: str = "adamw"
    lr: float = 1e-4
    num_classes: int = 100
    weight_decay: float = 0.01
    warmup: int = 1000
    total_steps: int = 10000
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    dtype: str = "bf16"
    epochs: int = 3
    logging_steps: int = 100
    save_steps: int = 1000
    save_dir: str = "./output"
    data_dir: str = "./data"
    dist_backend: str = "nccl"
    dist_url: str = "tcp://"
    dist_master_addr: str = ""
    dist_master_port: str = ""
    random_seed: int = 42
    world_size: int = 1
    global_rank: int = 0
    local_rank: int = 0
    device: str = "cuda"


class _MetaTrainer:
    def __init__(self, config):
        self.config = config

    def init_distributed(self):
        time.sleep(0.1 * self.config.local_rank)
        dist.init_process_group(
            backend=self.config.dist_backend,
            init_method=self.config.dist_url
            + f"{self.config.dist_master_addr}:"
            + f"{self.config.dist_master_port}",
            world_size=self.config.world_size,
            rank=self.config.global_rank,
        )
        torch.cuda.set_device(self.config.local_rank)
        import builtins as __builtin__
        builtin_print = __builtin__.print

        is_master = self.config.global_rank == 0

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

    def set_random_seed(self):
        if self.config.random_seed is not None:
            self.config.random_seed = time.time_ns() % 2 ** 32
        torch.manual_seed(self.config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)

    def worker(self, _idx):
        self.config.global_rank += _idx
        self.config.local_rank += _idx
        self.init_distributed()
        self.set_random_seed()
        self.log = Log(
            name=f"worker_{self.config.local_rank}",
            path=self.config.save_dir,
        )
        self.device = torch.device(f"cuda:{self.config.local_rank}")
