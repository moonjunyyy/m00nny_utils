import os
import torch
import ffmpeg
import numpy as np
from typing import Union, Iterable, Optional, Tuple
from .augmentation import Augmentation
from ..system.threads import Thread
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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)


class _ImageFile(_MediaFile):
    def __init__(self, filename):
        super().__init__(filename)
        # Get a single image stream info
        self._metadata = Thread(target=self._initialize_streams, daemon=True)
        self._metadata.start()

    def _initialize_streams(self):
        self.streams = [
            _stream
            for _stream in ffmpeg.probe(self.filename)["streams"]
            if _stream["codec_type"] == "video"
        ]
        self.widths = [int(_stream["width"]) for _stream in self.streams]
        self.heights = [int(_stream["height"]) for _stream in self.streams]

    def decode(
        self,
        request: "MediaDecodeRequest",
    ) -> Union[bytes, "np.ndarray", "torch.Tensor"]:
        if self._metadata is not None:
            self._metadata.join()
            self._metadata = None
        stream_idx = request.stream.index if request.stream else 0
        ffmpeg_stream = ffmpeg.input(self.filename)
        augments = request.aug_policy if request.aug_policy else None
        meta_data = {
                    "width": self.widths[stream_idx],
                    "height": self.heights[stream_idx],
                }
        if augments:
            ffmpeg_stream, meta_data = augments.apply(
                ffmpeg_stream,
                meta=meta_data
            )
        output, _ = ffmpeg_stream.output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
        ).run(capture_stdout=True, capture_stderr=True)
        if request.return_type == MediaFileReturnType.NumPy:
            output = self.bytes_to_ndarray(
                byte_data=output,
                shape=(
                    meta_data["height"],
                    meta_data["width"],
                    3,
                ),
                data_type=request.data_type,
                original_dtype=MediaFileDataType.Int8,
                normalize=request.normalize,
            )
        elif request.return_type == MediaFileReturnType.PyTorch:
            output = self.bytes_to_torch(
                byte_data=output,
                shape=(
                    meta_data["height"],
                    meta_data["width"],
                    3,
                ),
                data_type=request.data_type,
                original_dtype=MediaFileDataType.Int8,
                normalize=request.normalize,
            )
        return output


class _ImageFileManager(_MediaFileManager):
    _managed_class = _ImageFile

class Image(_Media):
    _manager_class = _ImageFileManager

    def __getitem__(
        self,
        aug_policy: Optional[Augmentation] = None,
        stream: Optional[MediaStreamRef] = None,
    ):
        request = MediaDecodeRequest(
            stream=stream,
            time_slice=None,
            return_type=self.return_type,
            data_type=self.data_type,
            normalize=self.normalize,
            aug_policy=aug_policy,
        )
        return self._file.decode(request)
