import torch
from torch import Tensor, nn
from re import Pattern
from typing import Union, List, Callable, TypeVar
# import torch.distributed as dist

_RegEx = Union[str, Pattern, List[Union[str, Pattern]]]
_Tensor = TypeVar('_Tensor', bound=Tensor)
_Module = TypeVar('_Module', bound=torch.nn.Module)
_Param = TypeVar('_Param', bound=torch.nn.Parameter)

_PreForwardHook = Callable[[_Module, _Tensor], None]
_ForwardHook = Callable[[_Module, _Tensor, _Tensor], None]
_BackwardHook = Callable[[_Module, _Tensor], None]


class _MetaPEFTHandler(nn.Module):
    def __init__(
        self,
        module: _Module,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.module = module
        self._enabled = True

    def enable(self): pass
    def disable(self): pass

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        if value:
            self.enable()
        else:
            self.disable()
