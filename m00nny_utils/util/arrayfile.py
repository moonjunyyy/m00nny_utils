import io
import gzip
import json
import zlib
import struct
import torch
import ffmpeg
import numpy as np
from typing import Dict, Hashable, Iterable, Optional, Union
from ..system.task_pool import Task
from .mediafile import (
    _Media,
    _MediaFile,
    _MediaFileManager,
    PathLike,
    MediaFileReturnType,
    MediaFileDataType,
    MediaStreamRef,
    MediaDecodeRequest,
)

# Containers: files that hold arrays, each read here from its own format into {key: ndarray}.
#   idx           the MNIST family (one array, key 0)
#   MAT-file v5   MATLAB .mat (numeric arrays by variable name)
#   safetensors   (arrays by tensor name)
#   torch         torch.save of tensors (a tensor: key 0; a tuple or list: its positions; a dict: its keys)
#   records       a file that is nothing but fixed-size records (CIFAR's binary version): the fields of a record
#                 are declared by the reader's caller, as (name, type[, shape]); arrays by field name
# A gzip-compressed container is read through.

_IDX_TYPES = {0x08: ">u1", 0x09: ">i1", 0x0B: ">i2", 0x0C: ">i4", 0x0D: ">f4", 0x0E: ">f8"}


def read_idx(data: bytes) -> Dict[Hashable, np.ndarray]:
    # 0 0 <type> <number of dimensions>, the dimensions (big-endian), the values
    ndim = data[3]
    dims = struct.unpack_from(">" + "I" * ndim, data, 4)
    return {0: np.frombuffer(data, _IDX_TYPES[data[2]], offset=4 + 4 * ndim).reshape(dims)}


# MAT-file v5: a 128-byte header, then data elements (tag: type, number of bytes; body; padding to 8 bytes).
_MAT_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 9: "f8", 12: "i8", 13: "u8"}
_MAT_CLASSES = {6: "f8", 7: "f4", 8: "i1", 9: "u1", 10: "i2", 11: "u2", 12: "i4", 13: "u4", 14: "i8", 15: "u8"}
_MAT_MATRIX = 14
_MAT_COMPRESSED = 15
_MAT_PEEK = 512   # bytes of a compressed element that hold its flags, dimensions and name


def _mat_element(data, pos, order):
    """(type, body, position of the next element) of the data element at pos."""
    first = struct.unpack_from(order + "I", data, pos)[0]
    if first >> 16:
        # Small element: the number of bytes and the type share the first word, the value follows in 4 bytes.
        return first & 0xFFFF, data[pos + 4:pos + 4 + (first >> 16)], pos + 8
    nbytes = struct.unpack_from(order + "I", data, pos + 4)[0]
    end = pos + 8 + nbytes
    return first, data[pos + 8:end], end if first == _MAT_COMPRESSED else end + (-nbytes) % 8


def _mat_matrix_head(body, order):
    """(class, dimensions, name, position of the values) of a matrix element."""
    _, flags, pos = _mat_element(body, 0, order)
    _, dims, pos = _mat_element(body, pos, order)
    _, name, pos = _mat_element(body, pos, order)
    return (struct.unpack_from(order + "I", flags, 0)[0] & 0xFF,
            np.frombuffer(dims, order + "i4").tolist(), bytes(name).decode("latin-1"), pos)


def _mat_matrix(body, order):
    mclass, dims, name, pos = _mat_matrix_head(body, order)
    if mclass not in _MAT_CLASSES:
        return name, None   # cell, struct, object, char, sparse: not an array of numbers
    mtype, real, _ = _mat_element(body, pos, order)
    # The values may be stored in a smaller type than the class of the matrix; column-major order.
    array = np.frombuffer(real, np.dtype(_MAT_TYPES[mtype]).newbyteorder(order))
    return name, array.astype(_MAT_CLASSES[mclass], copy=False).reshape(dims, order="F")


def read_mat(data: bytes, keys: Optional[Iterable[str]] = None) -> Dict[Hashable, np.ndarray]:
    if data[:10] == b"MATLAB 7.3":
        raise ValueError("MAT-file version 7.3 (HDF5) is not read here; version 5 is")
    order = "<" if data[126:128] == b"IM" else ">"
    keys = None if keys is None else set(keys)
    pos, arrays = 128, {}
    while pos + 8 <= len(data):
        mtype, body, pos = _mat_element(data, pos, order)
        if mtype == _MAT_COMPRESSED:
            if keys is not None:
                # The name is inside the compressed stream: unpack its beginning only, skip what is not asked for.
                head = zlib.decompressobj().decompress(body, _MAT_PEEK)
                try:
                    if _mat_matrix_head(head[8:], order)[2] not in keys:
                        continue
                except (struct.error, ValueError):
                    pass   # a name longer than the peek: unpack the whole element
            mtype, body, _ = _mat_element(zlib.decompress(body), 0, order)
        if mtype == _MAT_MATRIX:
            name, array = _mat_matrix(body, order)
            if array is not None and (keys is None or name in keys):
                arrays[name] = array
    return arrays


_SAFETENSORS_TYPES = {
    "BOOL": "?", "U8": "u1", "I8": "i1", "I16": "<i2", "U16": "<u2", "I32": "<i4", "U32": "<u4",
    "I64": "<i8", "U64": "<u8", "F16": "<f2", "F32": "<f4", "F64": "<f8",
}


def read_safetensors(data: bytes) -> Dict[Hashable, np.ndarray]:
    # The length of the header (8 bytes, little-endian), the header (JSON: name -> dtype, shape, data_offsets), the values.
    n = struct.unpack_from("<Q", data, 0)[0]
    header = json.loads(bytes(data[8:8 + n]))
    arrays = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, _ = info["data_offsets"]
        count = int(np.prod(info["shape"], dtype=np.int64))
        arrays[name] = np.frombuffer(
            data, _SAFETENSORS_TYPES[info["dtype"]], count=count, offset=8 + n + begin
        ).reshape(info["shape"])
    return arrays


def read_torch(data: bytes) -> Dict[Hashable, np.ndarray]:
    # torch reads its own format; weights_only: tensors and plain containers, nothing else is unpickled.
    obj = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if torch.is_tensor(obj):
        obj = (obj,)
    items = obj.items() if isinstance(obj, dict) else enumerate(obj)
    return {key: value.numpy() for key, value in items if torch.is_tensor(value)}


def read_records(data: bytes, fields) -> Dict[Hashable, np.ndarray]:
    records = np.frombuffer(data, np.dtype([tuple(field) for field in fields]))
    return {name: records[name] for name in records.dtype.names}


def _is_safetensors(data):
    if len(data) < 9:
        return False
    n = struct.unpack_from("<Q", data, 0)[0]
    return 0 < n <= len(data) - 8 and data[8:9] == b"{"


# (what the beginning of the file looks like, the reader); the first that matches reads it.
_CONTAINERS = (
    (lambda data: data[:6] == b"MATLAB", read_mat),
    (lambda data: data[:4] == b"PK\x03\x04", read_torch),
    (lambda data: data[:2] == b"\x00\x00" and data[2] in _IDX_TYPES, read_idx),
    (_is_safetensors, read_safetensors),
    (lambda data: data[:1] == b"\x80", read_torch),   # torch's format before it was a zip
)


def read_arrays(
    path: PathLike,
    keys: Optional[Iterable[Hashable]] = None,
    records=None,
) -> Dict[Hashable, np.ndarray]:
    """The arrays of a container file; keys: only these (a MAT-file then unpacks only these);
    records: the fields of one record, for a file of plain records (it has no header to tell its format by)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if records is not None:
        arrays = read_records(data, records)
        return arrays if keys is None else {key: array for key, array in arrays.items() if key in set(keys)}
    for looks_like, reader in _CONTAINERS:
        if looks_like(data):
            arrays = reader(data, keys) if reader is read_mat else reader(data)
            if keys is not None:
                arrays = {key: array for key, array in arrays.items() if key in set(keys)}
            return arrays
    raise ValueError(f"{path}: not a container format read here (idx, MAT-file v5, safetensors, torch)")


_PIX_FMT = {1: "gray", 3: "rgb24"}   # channels of a stored frame -> what ffmpeg calls it


class _ArrayFile(_MediaFile):
    _kind = "video"

    def __init__(self, filename, key=None, layout=None, records=None):
        super().__init__(filename)
        self.key = key
        self.layout = layout
        self.records = records
        # Reading the container is the probe.
        self._metadata = Task(target=self._initialize_streams, daemon=True)
        self._metadata.start()

    def _initialize_streams(self):
        arrays = read_arrays(self.filename, None if self.key is None else (self.key,), self.records)
        if not arrays:
            raise KeyError(f"{self.filename} has no array {self.key!r}")
        frames = arrays[next(iter(arrays)) if self.key is None else self.key]
        if frames.dtype != np.uint8:
            raise TypeError(f"{self.filename}: images are expected as uint8, not {frames.dtype}")
        # layout: the axes of the stored array, of N (images), H, W, C. By default N first, and the channels where
        # an axis of 1, 3 or 4 is (last, else second).
        layout = self.layout
        if layout is None:
            layout = "NHW" if frames.ndim == 3 else ("NHWC" if frames.shape[-1] in (1, 3, 4) else "NCHW")
        frames = frames.transpose([layout.index(axis) for axis in "NHWC" if axis in layout])
        if frames.ndim == 4 and frames.shape[3] == 1:
            frames = frames[..., 0]
        self.frames = frames   # (N, H, W) or (N, H, W, C)
        self.streams = [{"codec_type": "video"}]
        self.widths = [frames.shape[2]]
        self.heights = [frames.shape[1]]

    def decode(
        self,
        request: "MediaDecodeRequest",
    ) -> Union[bytes, "np.ndarray", "torch.Tensor"]:
        self._ready()
        index = request.time_slice
        if not isinstance(index, int):
            raise ValueError("Array time_slice must be the index of an image")
        frame = self.frames[index]
        channels = 1 if frame.ndim == 2 else frame.shape[2]
        augments = request.aug_policy
        meta_data = {
            "width": self.widths[0],
            "height": self.heights[0],
        }
        if augments:
            # The same filter graph as for a file, fed with the frame through a pipe.
            ffmpeg_stream = ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt=_PIX_FMT[channels],
                s=f"{meta_data['width']}x{meta_data['height']}",
            )
            ffmpeg_stream, meta_data = augments.apply(
                ffmpeg_stream,
                meta=meta_data
            )
            output = self._run(ffmpeg_stream.output(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
            ), input=frame.tobytes())
        else:
            # Nothing to filter: the frame as it is, in three channels as a decoded file is.
            if channels != 3:
                frame = np.repeat(frame.reshape(frame.shape[0], frame.shape[1], 1), 3, axis=2)
            output = frame.tobytes()
        if request.return_type == MediaFileReturnType.NumPy:
            output = self.bytes_to_ndarray(
                byte_data=output,
                shape=(meta_data["height"], meta_data["width"], 3),
                data_type=request.data_type,
                original_dtype=MediaFileDataType.UInt8,
                normalize=request.normalize,
            )
        elif request.return_type == MediaFileReturnType.PyTorch:
            output = self.bytes_to_torch(
                byte_data=output,
                shape=(meta_data["height"], meta_data["width"], 3),
                data_type=request.data_type,
                original_dtype=MediaFileDataType.UInt8,
                normalize=request.normalize,
            )
        return output


class _ArrayFileManager(_MediaFileManager):
    _managed_class = _ArrayFile


class ArrayImage(_Media):
    """One image of an array container: image[augment], as Image. The container is read once for all its images."""
    _manager_class = _ArrayFileManager

    def __init__(
        self,
        filename: PathLike,
        index: int,
        key: Optional[Hashable] = None,
        layout: Optional[str] = None,
        records=None,
        return_type: MediaFileReturnType = MediaFileReturnType.PyTorch,
        data_type: MediaFileDataType = MediaFileDataType.Float32,
        normalize=None,
    ):
        super().__init__(filename, return_type, data_type, normalize, file_args=(key, layout, records))
        self.index = index

    def __getitem__(self, key=None):
        # The image of this handle is the time slice of the request.
        return super().__getitem__((self.index,) + (key if isinstance(key, tuple) else (key,)))
