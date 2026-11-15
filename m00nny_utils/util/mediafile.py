import os
import numpy as np
import torch
from enum import Enum
from ..system.threads import Lock
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Iterable
from .augmentation import Augmentation
import ffmpeg

PathLike = Union[str, bytes, os.PathLike]
FFmpegStream = Iterable[ffmpeg.nodes.FilterableStream]


class MediaFileReturnType(Enum):
    NumPy = "ndarray"
    PyTorch = "torch"
    RawBytes = "bytes"


class MediaFileDataType(Enum):
    Int8 = "int8"
    Int16 = "int16"
    Int32 = "int32"
    Float16 = "float16"
    Float32 = "float32"
    Float64 = "float64"


@dataclass
class MediaStreamRef:
    index: int
    kind: str


@dataclass
class MediaDecodeRequest:
    stream: MediaStreamRef
    time_slice: Union[int, slice]
    return_type: MediaFileReturnType
    data_type: MediaFileDataType
    normalize: Optional[Tuple[float, float]] = None
    aug_policy: Optional[Augmentation] = None

_np_dtype_map = {
    MediaFileDataType.Int8: np.int8,
    MediaFileDataType.Int16: np.int16,
    MediaFileDataType.Int32: np.int32,
    MediaFileDataType.Float16: np.float16,
    MediaFileDataType.Float32: np.float32,
    MediaFileDataType.Float64: np.float64,
}
_torch_dtype_map = {
    MediaFileDataType.Int8: torch.int8,
    MediaFileDataType.Int16: torch.int16,
    MediaFileDataType.Int32: torch.int32,
    MediaFileDataType.Float16: torch.float16,
    MediaFileDataType.Float32: torch.float32,
    MediaFileDataType.Float64: torch.float64,
}

class _MediaFile:
    def __init__(
        self,
        filename: PathLike,
    ):
        """
        Represents a media file (audio or video) to load and process it.
        Args:
            filename (PathLike):
                Path to the media file.

            return_type (MediaFileReturnType):
                Desired return type for the media data.
                Options are NumPy array, PyTorch tensor, or raw bytes.
                Default is PyTorch tensor.

            data_type (MediaFileDataType):
                Desired data type for the media data.
                Options are int8, int16, int32, float16, float32, float64.
                Default is float32.

            normalize (Optional[Tuple[float, float]]):
                Normalization parameters (mean, std)
                to apply if return_type is PyTorch tensor.
                Default is None.
        """
        self.filename = filename

    def decode(
        self,
        request: MediaDecodeRequest,
    ) -> Union[bytes, "np.ndarray", "torch.Tensor"]:
        raise NotImplementedError(
            "MediaFile is an abstract base class. "
            "Please use AudioFile or VideoFile subclasses."
        )

    def get_streams(self) -> Iterable[MediaStreamRef]:
        raise NotImplementedError(
            "MediaFile is an abstract base class. "
            "Please use AudioFile or VideoFile subclasses."
        )

    def bytes_to_ndarray(
        self,
        byte_data: bytes,
        shape: Tuple[int, ...],
        data_type: MediaFileDataType,
        original_dtype: Optional[MediaFileDataType] = MediaFileDataType.Int8,
        normalize: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        np_dtype = _np_dtype_map[data_type]
        original_dtype = _np_dtype_map[original_dtype]
        array = np.frombuffer(byte_data, dtype=original_dtype)
        array = array.reshape(shape)
        return array

    def bytes_to_torch(
        self,
        byte_data: bytes,
        shape: Tuple[int, ...],
        data_type: MediaFileDataType,
        original_dtype: Optional[MediaFileDataType] = MediaFileDataType.Int8,
        normalize: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        torch_dtype = _torch_dtype_map[data_type]
        original_dtype = _torch_dtype_map[original_dtype]
        tensor = torch.frombuffer(byte_data, dtype=original_dtype)
        tensor = tensor.reshape(shape)
        return tensor


class _MediaFileManager:
    _managed_class = None

    def __new__(cls):
        if not hasattr(cls, "__instance"):
            cls._instance = super().__new__(cls)
            cls._instance._files = {}
            cls._instance._ref_count = {}
            cls._instance._lock = Lock()
        return cls._instance

    def get_file(self, filename):
        with self._lock:
            if filename not in self._files:
                self._files[filename] = self.__class__._managed_class(filename)
                self._ref_count[filename] = 0
            self._ref_count[filename] += 1
            return self._files[filename]

    def release_file(self, filename):
        with self._lock:
            if filename not in self._files:
                return
            self._ref_count[filename] -= 1
            if self._ref_count[filename] <= 0:
                del self._files[filename]
                del self._ref_count[filename]


class _Media:
    _manager_class = _MediaFileManager

    def __init__(
        self,
        filename: PathLike,
        return_type: MediaFileReturnType = MediaFileReturnType.PyTorch,
        data_type: MediaFileDataType = MediaFileDataType.Float32,
        normalize: Optional[Tuple[float, float]] = None,
    ):
        self.filename = filename
        self._manager = self.__class__._manager_class()
        self._file = self._manager.get_file(self.filename)
        self.return_type = return_type
        self.data_type = data_type
        self.normalize = normalize

    def get_stream(self) -> Iterable[MediaStreamRef]:
        self._file.get_stream_info()

    def __getitem__(
        self,
        idx: Union[int, slice],
        aug_policy: Optional[Augmentation] = None,
        stream: Optional[MediaStreamRef] = None,
    ):
        request = MediaDecodeRequest(
            stream=stream,
            time_slice=idx,
            return_type=self.return_type,
            data_type=self.data_type,
            normalize=self.normalize,
            aug_policy=aug_policy,
        )
        return self._file.decode(request)
