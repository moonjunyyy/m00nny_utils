import os
import torch
from typing import Dict, List, Optional, Tuple, Union

_Index = Union[int, slice, Tuple[Union[int, slice], ...]]

class Safetensor:
    """One .safetensors file. The header is read when the object is made; tensors are read from the file when
    asked for. The object does not change after that: one object can be read from several threads at once, and a
    read does not hold the GIL."""

    @property
    def path(self) -> str: ...

    def __init__(self, path: Union[str, "os.PathLike[str]"]) -> None: ...

    def keys(self) -> List[str]:
        """The names of the tensors, sorted (the file's "__metadata__" entry is not one)."""
        ...

    def info(self, name: str) -> Dict[str, object]:
        """What the header says of the tensor: "dtype" (the format's name, e.g. "F32"), "shape" (a tuple) and
        "data_offsets" (begin, end; from the start of the file's data section). KeyError if there is no such
        tensor."""
        ...

    def read(self, name: str, index: Optional[_Index] = None, *, dtype: Optional[torch.dtype] = None,
             device: Union[torch.device, str, int, None] = None) -> torch.Tensor:
        """The tensor, or the block of it `index` names as tensor[index] would: an int, a slice of step 1, or a
        tuple of them for the leading dimensions. Only the bytes of the block are read from the file.

        dtype: None keeps the dtype stored in the file. device: None is the CPU; an int is the index of a CUDA
        device.
        """
        ...
