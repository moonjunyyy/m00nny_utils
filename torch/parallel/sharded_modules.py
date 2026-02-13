import re
import json
from os import path
from typing import Tuple

import torch
import torch.distributed as dist
from transformers import AutoModel, AutoConfig
from safetensors import safe_open
from huggingface_hub import snapshot_download
from ...system.log import Log

log = Log(name="sharded_modules")


class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, op=dist.ReduceOp.SUM):
        ctx.op = op
        ctx.save_for_backward(input)
        dist.all_reduce(input, op=op)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # We need to divide the gradient by the world size
        op = ctx.op
        input = ctx.saved_tensors[0]
        if op == dist.ReduceOp.SUM:
            return grad_output, None
        elif op == dist.ReduceOp.PRODUCT:
            return grad_output / input, None
        elif op == dist.ReduceOp.AVG:
            return grad_output / dist.get_world_size(), None
        else:
            raise NotImplementedError(
                f"Impossible to perform backward for {op}")


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, strict_same_shape: bool = False):
        device = input.device
        dtype = input.dtype
        if not strict_same_shape:
            local_size = torch.tensor(input.size(), device=device)
            all_sizes = [
                torch.zeros_like(local_size)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(all_sizes, local_size)
            max_size = torch.max(torch.stack(all_sizes), dim=0).values
            padded = torch.empty(*max_size, device=device, dtype=dtype)
            slices = [slice(0, s) for s in input.size()]
            padded[slices] = input
        else:
            max_size = torch.tensor(input.size(), device=device)
            all_sizes = [max_size for _ in range(dist.get_world_size())]
            padded = input
        all_qs_padded = [torch.zeros_like(padded)
                         for _ in range(dist.get_world_size())]
        dist.all_gather(all_qs_padded, padded)
        if not strict_same_shape:
            all_qs = []
            for q, size in zip(all_qs_padded, all_sizes):
                size = [slice(0, s) for s in size]
                all_qs.append(q[size])
            return tuple(all_qs)
        else:
            return tuple(all_qs_padded)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs[dist.get_rank()], None


def all_reduce(
    input: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM
) -> torch.Tensor:
    if dist.is_initialized():
        return AllReduce.apply(input, op)
    else:
        return input


def all_gather(
    input: torch.Tensor, strict_same_shape: bool = False
) -> Tuple[torch.Tensor]:
    if dist.is_initialized():
        return AllGather.apply(input, strict_same_shape)
    else:
        return (input,)


class ShardedEmbedding(torch.nn.Module):
    def __init__(self, embedding: torch.nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        if dist.is_initialized():
            self.shard_vocab_size = self.num_embeddings // self.world_size + int(
                self.num_embeddings % self.world_size != 0
            )
            self.min_vocab_num = self.shard_vocab_size * self.rank
            self.max_vocab_num = min(
                self.shard_vocab_size * (self.rank + 1), self.num_embeddings
            )
        else:
            self.shard_vocab_size = None
            self.min_vocab_num = 0
            self.max_vocab_num = self.num_embeddings
        self.weight = torch.nn.Parameter(
            embedding.weight[self.min_vocab_num: self.max_vocab_num]
        )

    def forward(self, input):
        in_range = (input >= self.min_vocab_num) & (input < self.max_vocab_num)
        input = (input - self.min_vocab_num).masked_fill(in_range.logical_not(), 0)
        output = torch.nn.functional.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
        )
        output = output.masked_fill(in_range.logical_not().unsqueeze(-1), 0)
        if self.shard_vocab_size is not None:
            all_reduce(output, op=dist.ReduceOp.SUM)
        return output

    def extra_repr(self):
        repr = ""
        repr += f"num_embeddings={self.num_embeddings} "
        repr += f"({self.shard_vocab_size} / {self.world_size}), "
        repr += f"embedding_dim={self.embedding_dim}, "
        repr += f"padding_idx={self.padding_idx}, "
        repr += f"max_norm={self.max_norm}, "
        repr += f"norm_type={self.norm_type}, "
        repr += f"scale_grad_by_freq={self.scale_grad_by_freq}"
        return repr


class ShardedLinear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, row_parallel: bool = False):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        if dist.is_initialized():
            if row_parallel:
                self.shard_in_features = None
                self.shard_out_features = self.out_features // self.world_size + int(
                    self.out_features % self.world_size != 0
                )
                self.weight = torch.nn.Parameter(
                    linear.weight[
                        self.shard_out_features
                        * self.rank: self.shard_out_features
                        * (self.rank + 1)
                    ]
                )
                if self.bias:
                    self.bias = torch.nn.Parameter(
                        linear.bias[
                            self.shard_out_features
                            * self.rank: self.shard_out_features
                            * (self.rank + 1)
                        ]
                    )
            else:
                self.shard_in_features = self.in_features // self.world_size + int(
                    self.in_features % self.world_size != 0
                )
                self.shard_out_features = None
                self.weight = torch.nn.Parameter(
                    linear.weight[
                        :,
                        self.shard_in_features
                        * self.rank: self.shard_in_features
                        * (self.rank + 1),
                    ]
                )
                if self.bias:
                    self.bias = torch.nn.Parameter(linear.bias)
        else:
            self.weight = torch.nn.Parameter(linear.weight)
            if self.bias:
                self.bias = torch.nn.Parameter(linear.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.shard_in_features is not None:
            input = input[
                ...,
                self.shard_in_features
                * self.rank: self.shard_in_features
                * (self.rank + 1),
            ]
        output = torch.nn.functional.linear(input, self.weight, self.bias)
        if self.shard_in_features is not None:
            all_reduce(output, op=dist.ReduceOp.SUM)
        if self.shard_out_features is not None:
            output = torch.cat(all_gather(output), dim=-1)
        return output

    def extra_repr(self):
        repr = ""
        if self.shard_in_features is not None:
            repr += f"in_features={self.shard_in_features} * {self.world_size}, "
            repr += f"out_features={self.out_features}, "
        elif self.shard_out_features is not None:
            repr += f"in_features={self.in_features}, "
            repr += f"out_features={self.shard_out_features} * {self.world_size}, "
        else:
            repr += f"in_features={self.in_features}, "
            repr += f"out_features={self.out_features}, "
        repr += f"bias={self.bias is not None}"
        return repr


class shardedConv1D(torch.nn.Module):
    def __init__(self, conv1d: torch.nn.Conv1d, row_parallel: bool = False):
        super().__init__()
        """
        Sharded 1D Convolutional layer.
        Args:
            conv1d (torch.nn.Conv1d):
                The Conv1d layer to be sharded.

            row_parallel (bool, ignored):
                It is technically possible to shard Conv1D
                in both row and column parallel ways,
                but there is small benefit to shard Conv1D in row parallel way.
                Therefore, this argument is ignored.
        """
        self.in_channels = conv1d.in_channels
        self.out_channels = conv1d.out_channels
        self.kernel_size = conv1d.kernel_size
        self.stride = conv1d.stride
        self.padding = conv1d.padding
        self.dilation = conv1d.dilation
        self.groups = conv1d.groups
        self.bias = conv1d.bias

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        if dist.is_initialized():
            if row_parallel:
                log.warning(
                    "Sharding Conv1D in row parallel way is not recommended. "
                    "The argument 'row_parallel' is ignored."
                )
            self.shard_in_channels = self.in_channels // self.world_size + int(
                self.in_channels % self.world_size != 0
            )
            self.shard_out_channels = None
            self.weight = torch.nn.Parameter(
                conv1d.weight[
                    :,
                    :,
                    self.shard_in_channels
                    * self.rank: self.shard_in_channels
                    * (self.rank + 1),
                ]
            )
            if self.bias:
                self.bias = torch.nn.Parameter(conv1d.bias)
        else:
            self.weight = torch.nn.Parameter(conv1d.weight)
            if self.bias:
                self.bias = torch.nn.Parameter(conv1d.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.nn.functional.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.shard_out_channels is not None:
            output = torch.cat(all_gather(output), dim=1)
        return output


class shardedConv2D(torch.nn.Module):
    def __init__(self, conv2d: torch.nn.Conv2d, row_parallel: bool = False):
        super().__init__()
        """
        Sharded 2D Convolutional layer.
        Args:
            conv2d (torch.nn.Conv2d):
                The Conv2d layer to be sharded.

            row_parallel (bool, ignored):
                It is technically possible to shard Conv2D
                in both row and column parallel ways,
                but there is small benefit to shard Conv2D in row parallel way.
                Therefore, this argument is ignored.
        """
        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        self.kernel_size = conv2d.kernel_size
        self.stride = conv2d.stride
        self.padding = conv2d.padding
        self.dilation = conv2d.dilation
        self.groups = conv2d.groups
        self.bias = conv2d.bias

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        if dist.is_initialized():
            if row_parallel:
                log.warning(
                    "Sharding Conv2D in row parallel way is not recommended. "
                    "The argument 'row_parallel' is ignored."
                )
            self.shard_in_channels = self.in_channels // self.world_size + int(
                self.in_channels % self.world_size != 0
            )
            self.shard_out_channels = None
            self.weight = torch.nn.Parameter(
                conv2d.weight[
                    :,
                    :,
                    :,
                    self.shard_in_channels
                    * self.rank: self.shard_in_channels
                    * (self.rank + 1),
                ]
            )
            if self.bias:
                self.bias = torch.nn.Parameter(conv2d.bias)
        else:
            self.weight = torch.nn.Parameter(conv2d.weight)
            if self.bias:
                self.bias = torch.nn.Parameter(conv2d.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.nn.functional.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.shard_out_channels is not None:
            output = torch.cat(all_gather(output), dim=1)
        return output


def _set_module(name: str, module: torch.nn.Module, new_module: torch.nn.Module):
    names = name.split(".")
    sub_module = module
    for n in names[:-1]:
        sub_module = getattr(sub_module, n)
    setattr(sub_module, names[-1], new_module)


def convert_to_sharded_module(
    module: torch.nn.Module,
    embed_parallel_ids: list[str] = [],
    row_parallel_ids: list[str] = [],
    col_parallel_ids: list[str] = [],
    seq_parallel_ids: list[str] = [],
    conv_parallel_ids: list[str] = [],
) -> torch.nn.Module:
    for name, child in module.named_children():
        if any(name in _id for _id in embed_parallel_ids) and isinstance(
            child, torch.nn.Embedding
        ):
            setattr(module, name, ShardedEmbedding(child))
        elif any(name in _id for _id in row_parallel_ids) and isinstance(
            child, torch.nn.Linear
        ):
            setattr(module, name, ShardedLinear(child, row_parallel=True))
        elif any(name in _id for _id in col_parallel_ids) and isinstance(
            child, torch.nn.Linear
        ):
            setattr(module, name, ShardedLinear(child, row_parallel=False))
        elif any(name in _id for _id in conv_parallel_ids) and isinstance(
            child, torch.nn.Conv1d
        ):
            setattr(module, name, shardedConv1D(child))
        elif any(name in _id for _id in conv_parallel_ids) and isinstance(
            child, torch.nn.Conv2d
        ):
            setattr(module, name, shardedConv2D(child))


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
                shardedConv1D(module),
            )
        elif (
            any(re.match(_expression, module_name,)
                for _expression in conv_parallel_ids) and
            isinstance(module, torch.nn.Conv2d)
        ):
            _set_module(
                module_name,
                model,
                shardedConv2D(module),
            )
    return model


def load_transformers_as_sharded_module(
    model_name_or_path: str,
    embed_parallel_ids: list[str] = [],
    row_parallel_ids: list[str] = [],
    col_parallel_ids: list[str] = [],
    seq_parallel_ids: list[str] = [],
    conv_parallel_ids: list[str] = [],
    **kwargs,
) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
    model = AutoModel.from_config(config, device_map="meta")
    weight_location = snapshot_download(
        repo_id=model_name_or_path
    )

    state_dict = {}
    metadata = json.load(
        open(f"{path.join(weight_location, "model.safetensors.index.json")}"))

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    model = _convert_to_sharded_module_recursive(
        model,
        embed_parallel_ids,
        row_parallel_ids,
        col_parallel_ids,
        seq_parallel_ids,
        conv_parallel_ids,
    )

    for name, param in model.named_parameters():
        mode = 0
        if any(name in _id for _id in embed_parallel_ids):
            mode = 3
        elif any(name in _id for _id in row_parallel_ids):
            mode = 1
        elif any(name in _id for _id in col_parallel_ids):
            mode = 2

        weight_filename = metadata["weight_map"][name]
        if weight_filename is None:
            print(f"Weight file not found for {name} in metadata")
            continue
        with safe_open(
            path.join(weight_location, weight_filename),
            framework="pt",
            device=rank
        ) as f:
            if mode == 0:
                t = f.get_tensor(name)
            else:
                tensor_slice = f.get_slice(name)
                dim_out, dim_in = tensor_slice.get_shape()
                if mode == 1:
                    dim_slice = dim_in // world_size
                    t = tensor_slice[:, dim_slice *
                                     rank: dim_slice * (rank + 1)]
                elif mode == 2:
                    dim_slice = dim_out // world_size
                    t = tensor_slice[dim_slice *
                                     rank: dim_slice * (rank + 1), :]
                elif mode == 3:
                    t = tensor_slice[
                        model.language_model.model.embed_tokens.min_vocab_num:
                        model.language_model.model.embed_tokens.max_vocab_num,
                        :]
            if t.dtype != param.dtype:
                t = t.to(dtype=param.dtype)  # Convert to the parameter's dtype
            state_dict[name] = t
        model.load_state_dict(state_dict, assign=True)
        return model
