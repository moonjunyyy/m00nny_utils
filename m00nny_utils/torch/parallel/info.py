# What a process is told about its place among the others, and which modules
# are split across them. Apart from the loader that uses them, so that the
# runtime reads them with torch alone.
from dataclasses import dataclass

import torch


@dataclass
class MultiprocessDeviceInfo:
    global_world_size: int
    global_rank: int
    local_world_size: int
    local_rank: int
    device: torch.device


@dataclass
class TensorParallelInfo:
    embed_parallel_ids: list[str]
    row_parallel_ids: list[str]
    col_parallel_ids: list[str]
    seq_parallel_ids: list[str]
    conv_parallel_ids: list[str]
