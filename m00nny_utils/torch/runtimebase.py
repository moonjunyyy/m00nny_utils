import os
import time
import random
import importlib.util

import torch
import torch.distributed as dist

from ..system.log import Log
from .parallel.info import (
    TensorParallelInfo,
    MultiprocessDeviceInfo
)


class RuntimeBase:
    # The defaults are those util/configbase.py declares for the same names.
    def __init__(
        self,
        mode="train",
        epochs=100,
        batchsize=64,
        device="cuda",
        dtype=torch.float32,
        random_seed=None,
        num_workers=4,
        save_path="./output",
        dist={},
        opt={},
        lrs={},
        data={},
        *args,
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
            device=_dev,
        )
        time.sleep(0.1 * _l_rank)  # Stagger the initialization

        # the config names it "protocol"; "url" is the earlier key
        _dist_url = self.dist_config.get(
            "protocol", self.dist_config.get("url", "tcp://"))
        _master_addr = self.dist_config.get("master_addr", "localhost")
        _master_port = self.dist_config.get("master_port", "42355")
        _dist_url = f"{_dist_url}{_master_addr}:{_master_port}"
        _dist_bkend = self.dist_config.get("backend", "nccl")

        dist.init_process_group(
            backend=_dist_bkend,
            init_method=_dist_url,
            world_size=_g_world,
            rank=_g_rank,
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(_l_rank)

        import builtins as __builtin__
        builtin_print = __builtin__.print
        is_master = _g_rank == 0

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                # the log is created after this function returns
                if not hasattr(self, "log"):
                    return builtin_print(*args, **kwargs)
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
        random.seed(self.random_seed)
        # numpy's generator too, where numpy is installed: torch does not
        # require it
        if importlib.util.find_spec("numpy") is not None:
            importlib.import_module("numpy").random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    def run(self):
        procs = []
        for idx in range(1, self.dist_config["local_world_size"]):
            p = torch.multiprocessing.Process(target=self._entry, args=(idx,))
            p.start()
            procs.append(p)
        self._entry(0)
        for p in procs:
            p.join()

    def _entry(self, _idx):
        # every process: the shared preparation (worker), then the worker of
        # the mode, `_{mode}_worker`, when the subclass has one
        self.worker(_idx)
        _mode_worker = getattr(self, f"_{self.mode}_worker", None)
        if _mode_worker is not None:
            _mode_worker(_idx)

    def worker(self, _idx):
        self.dist_config["global_rank"] += _idx
        self.dist_config["local_rank"] = _idx
        self.init_distributed()
        self.log = Log(
            name=f"worker_{self.device_info.local_rank}",
            path=self.save_path,
            level=(
                Log.DEBUG if self.device_info.local_rank == 0 else Log.WARNING
            ),
            use_STDOUT=(self.device_info.local_rank == 0),
            use_STDERR=False,
        )
        self.log.info(f"Worker {self.device_info.local_rank} initialized.")
        self.set_random_seed()
