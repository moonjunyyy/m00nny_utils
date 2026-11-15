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
    UInt8 = "uint8"
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
    MediaFileDataType.UInt8: np.uint8,
    MediaFileDataType.Int8: np.int8,
    MediaFileDataType.Int16: np.int16,
    MediaFileDataType.Int32: np.int32,
    MediaFileDataType.Float16: np.float16,
    MediaFileDataType.Float32: np.float32,
    MediaFileDataType.Float64: np.float64,
}
_torch_dtype_map = {
    MediaFileDataType.UInt8: torch.uint8,
    MediaFileDataType.Int8: torch.int8,
    MediaFileDataType.Int16: torch.int16,
    MediaFileDataType.Int32: torch.int32,
    MediaFileDataType.Float16: torch.float16,
    MediaFileDataType.Float32: torch.float32,
    MediaFileDataType.Float64: torch.float64,
}

class _MediaFile:
    # What a stream of this file is called in a MediaStreamRef ("video", "audio"); set by the subclass.
    _kind = None

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
        # The subclass starts a thread that probes the streams and keeps it here until it has been joined.
        self._metadata = None

    def _ready(self) -> None:
        """Wait for the stream probe started at construction. Decoding threads may arrive here together."""
        metadata = self._metadata
        if metadata is not None:
            metadata.join()
            self._metadata = None

    def _run(self, output, input: Optional[bytes] = None) -> bytes:
        """Run an ffmpeg output node and return what it wrote to its pipe; input: what to feed its input pipe.
        -nostdin (when nothing is fed): ffmpeg otherwise reads the process's stdin for its interactive keys, i.e.
        the terminal of the program that is decoding."""
        if input is None:
            output = output.global_args("-nostdin")
        out, _ = output.run(capture_stdout=True, capture_stderr=True, input=input)
        return out

    def decode(
        self,
        request: MediaDecodeRequest,
    ) -> Union[bytes, "np.ndarray", "torch.Tensor"]:
        raise NotImplementedError(
            "MediaFile is an abstract base class. "
            "Please use AudioFile or VideoFile subclasses."
        )

    def get_streams(self) -> Iterable[MediaStreamRef]:
        self._ready()
        for idx, _ in enumerate(self.streams):
            yield MediaStreamRef(kind=self.__class__._kind, index=idx)

    def bytes_to_ndarray(
        self,
        byte_data: bytes,
        shape: Tuple[int, ...],
        data_type: MediaFileDataType,
        original_dtype: Optional[MediaFileDataType] = MediaFileDataType.UInt8,
        normalize: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        np_dtype = _np_dtype_map[data_type]
        original_dtype = _np_dtype_map[original_dtype]
        array = np.frombuffer(byte_data, dtype=original_dtype)
        array = array.reshape(shape)
        if normalize is not None:
            mean, std = normalize
            array = (array.astype(np.float64) - mean) / std
        return array.astype(np_dtype)

    def bytes_to_torch(
        self,
        byte_data: bytes,
        shape: Tuple[int, ...],
        data_type: MediaFileDataType,
        original_dtype: Optional[MediaFileDataType] = MediaFileDataType.UInt8,
        normalize: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        torch_dtype = _torch_dtype_map[data_type]
        original_dtype = _torch_dtype_map[original_dtype]
        # bytearray: torch.frombuffer wants a writable buffer (bytes is not); the result is this tensor's own memory.
        tensor = torch.frombuffer(bytearray(byte_data), dtype=original_dtype)
        tensor = tensor.reshape(shape)
        if normalize is not None:
            mean, std = normalize
            tensor = (tensor.to(torch.float64) - mean) / std
        return tensor.to(torch_dtype)


class _MediaFileManager:
    _managed_class = None

    def __new__(cls):
        # One manager per manager class (image, video, audio): looked up on the class itself, not inherited.
        if "_instance" not in cls.__dict__:
            cls._instance = super().__new__(cls)
            cls._instance._files = {}
            cls._instance._ref_count = {}
            cls._instance._lock = Lock()
        return cls._instance

    def get_file(self, filename, *args):
        # args: what the file object takes besides its name; a file opened with other args is another object.
        key = (filename, *args) if args else filename
        with self._lock:
            if key not in self._files:
                self._files[key] = self.__class__._managed_class(filename, *args)
                self._ref_count[key] = 0
            self._ref_count[key] += 1
            return self._files[key]

    def release_file(self, filename, *args):
        key = (filename, *args) if args else filename
        with self._lock:
            if key not in self._files:
                return
            self._ref_count[key] -= 1
            if self._ref_count[key] <= 0:
                del self._files[key]
                del self._ref_count[key]


class _Media:
    _manager_class = _MediaFileManager

    def __init__(
        self,
        filename: PathLike,
        return_type: MediaFileReturnType = MediaFileReturnType.PyTorch,
        data_type: MediaFileDataType = MediaFileDataType.Float32,
        normalize: Optional[Tuple[float, float]] = None,
        file_args: tuple = (),
    ):
        self.filename = filename
        self._file_args = tuple(file_args)
        self._manager = self.__class__._manager_class()
        self._file = self._manager.get_file(self.filename, *self._file_args)
        self.return_type = return_type
        self.data_type = data_type
        self.normalize = normalize

    def __del__(self):
        manager = getattr(self, "_manager", None)
        if manager is not None:
            manager.release_file(self.filename, *getattr(self, "_file_args", ()))

    def get_stream(self) -> Iterable[MediaStreamRef]:
        return self._file.get_streams()

    def __getitem__(
        self,
        key: Union[None, int, slice, Augmentation, MediaStreamRef, tuple] = None,
    ):
        """
        Decode the media. The key says what to decode and how, by the type of each part, alone or
        several in a tuple (any order):
            int or slice     the time slice (seconds); none: the whole stream
            Augmentation     the augmentation policy applied while decoding
            MediaStreamRef   the stream; none: the first
        image[augment], video[0:3], video[0:3, augment], audio[1:2, augment, stream], image[None]
        """
        idx, aug_policy, stream = None, None, None
        for part in key if isinstance(key, tuple) else (key,):
            if part is None:
                continue
            if isinstance(part, Augmentation):
                aug_policy = part
            elif isinstance(part, MediaStreamRef):
                stream = part
            elif isinstance(part, (int, slice)):
                idx = part
            else:
                raise TypeError(
                    f"{self.__class__.__name__}[...]: a time slice, an Augmentation or a MediaStreamRef "
                    f"is expected, not {type(part).__name__}"
                )
        request = MediaDecodeRequest(
            stream=stream,
            time_slice=idx,
            return_type=self.return_type,
            data_type=self.data_type,
            normalize=self.normalize,
            aug_policy=aug_policy,
        )
        return self._file.decode(request)
