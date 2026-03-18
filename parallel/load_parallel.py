import re
import json
from os import path

from dataclasses import dataclass
import torch
import torch.distributed as dist
from safetensors import safe_open
from huggingface_hub import snapshot_download
from ..system.log import Log
from ..system.task_pool import Task
from .sharded_modules import (
    ShardedEmbedding,
    ShardedLinear,
    ShardedConv1D,
    ShardedConv2D
)

log = Log(name="load_parallel")


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


def _set_module(
    name: str,
    module: torch.nn.Module,
    new_module: torch.nn.Module
):
    names = name.split(".")
    sub_module = module
    for n in names[:-1]:
        sub_module = getattr(sub_module, n)
    setattr(sub_module, names[-1], new_module)


def _check_regex_match(name: str, regex_list: list[str]) -> bool:
    for regex in regex_list:
        if re.match(regex, name):
            return True
    return False


def convert_to_sharded_module(
    module: torch.nn.Module,
    parallel_info: TensorParallelInfo
) -> torch.nn.Module:
    embed_parallel_ids = parallel_info.embed_parallel_ids
    row_parallel_ids = parallel_info.row_parallel_ids
    col_parallel_ids = parallel_info.col_parallel_ids
    conv_parallel_ids = parallel_info.conv_parallel_ids

    # The sequeance parallel, adopted to the normalization layer
    # can be model dependant, so it is not implemented here.
    # seq_parallel_ids = parallel_info.seq_parallel_ids

    for name, child in module.named_children():
        if _check_regex_match(
            name, embed_parallel_ids
        ) and isinstance(child, torch.nn.Embedding):
            setattr(module, name, ShardedEmbedding(child))
        elif _check_regex_match(
            name, row_parallel_ids
        ) and isinstance(child, torch.nn.Linear):
            setattr(module, name, ShardedLinear(child, row_parallel=True))
        elif _check_regex_match(
            name, col_parallel_ids
        ) and isinstance(child, torch.nn.Linear):
            setattr(module, name, ShardedLinear(child, row_parallel=False))
        elif _check_regex_match(
            name, conv_parallel_ids
        ) and isinstance(child, torch.nn.Conv1d):
            setattr(module, name, ShardedConv1D(child))
        elif _check_regex_match(
            name, conv_parallel_ids
        ) and isinstance(child, torch.nn.Conv2d):
            setattr(module, name, ShardedConv2D(child))


def _convert_to_sharded_module_recursive(
    model: torch.nn.Module,
    embed_parallel_ids: list[str] = [],
    row_parallel_ids: list[str] = [],
    col_parallel_ids: list[str] = [],
    seq_parallel_ids: list[str] = [],
    conv_parallel_ids: list[str] = [],
    prefix: str = "",
) -> torch.nn.Module:
    for module_name, module in model.named_modules():
        if (
            any(re.match(_expression, module_name,)
                for _expression in embed_parallel_ids) and
            isinstance(module, torch.nn.Embedding)
        ):
            _set_module(
                module_name,
                model,
                ShardedEmbedding(module),
            )
        elif (
            any(re.match(_expression, module_name,)
                for _expression in row_parallel_ids) and
            isinstance(module, torch.nn.Linear)
        ):
            _set_module(
                module_name,
                model,
                ShardedLinear(module, row_parallel=True),
            )
        elif (
            any(re.match(_expression, module_name,)
                for _expression in col_parallel_ids) and
            isinstance(module, torch.nn.Linear)
        ):
            _set_module(
                module_name,
                model,
                ShardedLinear(module, row_parallel=False),
            )
        elif (
            any(re.match(_expression, module_name,)
                for _expression in conv_parallel_ids) and
            isinstance(module, torch.nn.Conv1d)
        ):
            _set_module(
                module_name,
                model,
                ShardedConv1D(module),
            )
        elif (
            any(re.match(_expression, module_name,)
                for _expression in conv_parallel_ids) and
            isinstance(module, torch.nn.Conv2d)
        ):
            _set_module(
                module_name,
                model,
                ShardedConv2D(module),
            )
    return model


def _load_single_tensor_shareded(
    name: str,
    path: str,
    mode: int,
    num_shard: int,
    pos_shard: int,
    device_id: int,
    dtype: torch.dtype
) -> torch.Tensor:
    t = None
    with safe_open(
        path,
        framework="pt",
        device=device_id
    ) as f:
        if mode == 0:
            t = f.get_tensor(name)
        else:
            tensor_slice = f.get_slice(name)
            dim = tensor_slice.get_shape()
            if 'weight' in name:
                dim_in = dim[1]
                dim_out = dim[0]
                if mode == 1:
                    dim_slice = dim_in // num_shard
                    t = tensor_slice[
                        :, dim_slice * pos_shard: dim_slice * (pos_shard + 1)]
                elif mode == 2:
                    dim_slice = dim_out // num_shard
                    t = tensor_slice[
                        dim_slice * pos_shard: dim_slice * (pos_shard + 1), :]
                elif mode == 3:
                    dim_slice = (
                        dim_out // num_shard +
                        int(dim_out % num_shard != 0)
                    )
                    t = tensor_slice[
                        dim_slice * pos_shard: dim_slice * (pos_shard + 1), :]
            else:
                dim_out = dim[0]
                if mode == 1:
                    t = f.get_tensor(name)
                elif mode == 2:
                    dim_slice = dim_out // num_shard
                    t = tensor_slice[
                        dim_slice * pos_shard: dim_slice * (pos_shard + 1)]
    if t is not None and t.dtype != dtype:
        t = t.to(dtype=dtype)  # Convert to the parameter's dtype
    return t


def load_transformers_as_sharded_module(
    model,
    model_name_or_path: str,
    parallel_info: TensorParallelInfo,
    device_info: MultiprocessDeviceInfo,
    tp_depth: int = 1,
    **kwargs,
) -> torch.nn.Module:
    weight_location = snapshot_download(repo_id=model_name_or_path)
    state_dict = {}
    metadata = json.load(
        open(f"{path.join(weight_location, 'model.safetensors.index.json')}")
    )
    model = convert_to_sharded_module(model, parallel_info)
    num_shard = tp_depth
    pos_shard = device_info.local_rank % tp_depth
    device_id = device_info.local_rank

    for name, param in model.named_parameters():
        mode = 0
        if _check_regex_match(name, parallel_info.embed_parallel_ids):
            mode = 3
        elif _check_regex_match(name, parallel_info.row_parallel_ids):
            mode = 1
        elif _check_regex_match(name, parallel_info.col_parallel_ids):
            mode = 2
        weight_filename = metadata["weight_map"][name]
        if weight_filename is None:
            print(f"Weight file not found for {name} in metadata")
            continue
        state_dict[name] = Task(
            target=_load_single_tensor_shareded,
            kwargs={
                "name": name,
                "path": f"{weight_location}/{weight_filename}",
                "mode": mode,
                "num_shard": num_shard,
                "pos_shard": pos_shard,
                "device_id": device_id,
                "dtype": param.dtype,
            }
        )
    for k, v in state_dict.items():
        if v is None:
            print(
                f"Tensor for {k} is None, which may cause error when loading.")
        state_dict[k] = v.join()
    model.load_state_dict(state_dict, assign=True)
    return model
