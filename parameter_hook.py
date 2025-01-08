import torch
import torch.distributed as dist
from functools import partial

class ParameterHook:
    def __init__(self, module):
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._params = []
        self._grad_buffers = []
        self._hook_handles = []
        self._reduce_handles = []
        self._post_accumulate_grad_hook_handles = []
        _index = 0
        for name, param in module.named_parameters():
            if param.requires_grad:
                self._params.append(param)
                self._reduce_handles.append(None)
                self._grad_buffers.append(torch.zeros_like(param))
                self._hook_handles.append(param.register_hook(partial(self._hook, _index)))
                self._post_accumulate_grad_hook_handles.append(param.register_post_accumulate_grad_hook(partial(self._join, _index)))
                _index += 1

    def _hook(self, index, grad):
        self._grad_buffers[index].copy_(grad)
        _handle = dist.all_reduce(self._grad_buffers[index], async_op=True)
        self._reduce_handles[index] = _handle
        return grad

    def _join(self, index, *args):
        while self._reduce_handles[index] is None: continue
        self._reduce_handles[index].wait()
        self._reduce_handles[index] = None
        self._params[index].grad.copy_(self._grad_buffers[index])
        self._grad_buffers[index].zero_()

    def __del__(self):
        for handle in self._hook_handles: handle.remove()
        for handle in self._post_accumulate_grad_hook_handles: handle.remove()