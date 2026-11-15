import os
import torch
import ffmpeg
import numpy as np
from fractions import Fraction
from ..system.task_pool import Task
from typing import Union, Iterable
from .mediafile import (
    _Media,
    _MediaFile,
    _MediaFileManager,
    PathLike,
    MediaFileReturnType,
    MediaStreamRef,
    MediaDecodeRequest,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)


class _VideoFile(_MediaFile):
    _kind = "video"

    def __init__(self, filename):
        super().__init__(filename)
        self._metadata = Task(target=self._initialize_streams, daemon=True)
        self._metadata.start()

    def _initialize_streams(self):
        self.streams = [
            _stream
            for _stream in ffmpeg.probe(self.filename)["streams"]
            if _stream["codec_type"] == "video"
        ]
        self.widths = [int(_stream["width"]) for _stream in self.streams]
        self.heights = [int(_stream["height"]) for _stream in self.streams]
        self.frame_rates = [_stream["r_frame_rate"] for _stream in self.streams]
        self.duration = [float(_stream["duration"]) for _stream in self.streams]
        # Containers that do not store the frame count: duration x frame rate.
        self.nb_frames = [
            int(_stream["nb_frames"]) if "nb_frames" in _stream
            else int(round(float(_stream["duration"]) * float(Fraction(_stream["r_frame_rate"]))))
            for _stream in self.streams
        ]
        self.tags = [_stream.get("tags", {}) for _stream in self.streams]

    def decode(
        self,
        request: "MediaDecodeRequest",
    ) -> Union[bytes, "np.ndarray", "torch.Tensor"]:
        self._ready()
        stream_idx = request.stream.index if request.stream else 0
        if stream_idx < 0 or stream_idx >= len(self.streams):
            raise IndexError("Video stream index out of range")
        time_slice = request.time_slice
        if time_slice is None:
            time_slice = slice(None)  # the whole stream
        if not isinstance(time_slice, slice):
            raise ValueError("Video time_slice must be a slice object")
        start = time_slice.start or 0
        stop = time_slice.stop or self.duration[stream_idx]
        if start < 0 or stop > self.duration[stream_idx] or start >= stop:
            raise IndexError("Time slice out of range")
        start_time = start
        duration = stop - start
        ffmpeg_stream = ffmpeg.input(self.filename, ss=start_time, t=duration)
        augments = request.aug_policy
        frame_rate = float(Fraction(self.frame_rates[stream_idx]))
        # The metadata describes what is decoded: the slice, not the whole stream.
        meta_data = {
            "width": self.widths[stream_idx],
            "height": self.heights[stream_idx],
            "duration": duration,
            "nb_frames": int(round(duration * frame_rate)),
            "frame_rate": frame_rate,
        }
        if augments:
            ffmpeg_stream, meta_data = augments.apply(
                ffmpeg_stream,
                meta=meta_data,
            )
        output = self._run(ffmpeg_stream.output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
        ))
        width = meta_data.get("width", self.widths[stream_idx])
        height = meta_data.get("height", self.heights[stream_idx])
        # The frame count is what the decoder produced (the metadata's count is an estimate at slice borders).
        n_frames = len(output) // (height * width * 3)
        if request.return_type == MediaFileReturnType.NumPy:
            output = self.bytes_to_ndarray(
                output,
                shape=(n_frames, height, width, 3),
                data_type=request.data_type,
                normalize=request.normalize,
            )
        elif request.return_type == MediaFileReturnType.PyTorch:
            output = self.bytes_to_torch(
                output,
                shape=(n_frames, height, width, 3),
                data_type=request.data_type,
                normalize=request.normalize,
            )
        return output


class _VideoFileManager(_MediaFileManager):
    _managed_class = _VideoFile


class Video(_Media):
    _manager_class = _VideoFileManager


# The earlier name of Video.
VideoFile = Video
