import torch
import ffmpeg
import numpy as np
from ..system.threads import Thread
from typing import Union
from .mediafile import (
    _Media,
    _MediaFile,
    _MediaFileManager,
    MediaDecodeRequest,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406],
                         dtype=np.float32).reshape(1, 1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225],
                        dtype=np.float32).reshape(1, 1, 1, 3)


class _VideoFile(_MediaFile):
    def __init__(self, filename):
        super().__init__(filename)
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
        self.frame_rates = [_stream["r_frame_rate"]
                            for _stream in self.streams]
        self.nb_frames = [int(_stream["nb_frames"])
                          for _stream in self.streams]
        self.duration = [float(_stream["duration"])
                         for _stream in self.streams]
        self.tags = [_stream.get("tags", {}) for _stream in self.streams]

    def decode(
        self,
        request: "MediaDecodeRequest",
    ) -> Union[bytes, "np.ndarray", "torch.Tensor"]:
        if self._metadata is not None:
            self._metadata.join()
            self._metadata = None
        stream_idx = request.stream.index if request.stream else 0
        if stream_idx < 0 or stream_idx >= len(self.streams):
            raise IndexError("Video stream index out of range")
        slice = request.time_slice
        if not isinstance(slice, slice):
            raise ValueError("Video time_slice must be a slice object")
        start = slice.start or 0
        stop = slice.stop or self.duration[stream_idx]
        if start < 0 or stop > self.duration[stream_idx] or start >= stop:
            raise IndexError("Time slice out of range")
        start_time = start
        duration = stop - start
        ffmpeg_stream = ffmpeg.input(self.filename, ss=start_time, t=duration)
        augments = request.augments if request.stream else None
        meta_data = {
            "width": self.widths[stream_idx],
            "height": self.heights[stream_idx],
            "duration": self.duration[stream_idx],
            "nb_frames": self.nb_frames[stream_idx],
            "frame_rate": eval(self.frame_rates[stream_idx]),
        }
        if augments:
            ffmpeg_stream, meta_data = augments.apply(
                ffmpeg_stream,
                meta=meta_data,
            )
        output, _ = ffmpeg_stream.output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
        ).run(capture_stdout=True, capture_stderr=True)
        n_frames = meta_data.get("nb_frames", self.nb_frames[stream_idx])
        width = meta_data.get("width", self.widths[stream_idx])
        height = meta_data.get("height", self.heights[stream_idx])
        if request.return_type == "ndarray":
            output = self.bytes_to_ndarray(
                output,
                shape=(n_frames, height, width, 3),
                data_type=request.data_type,
                normalize=request.normalize,
            )
        elif request.return_type == "torch":
            output = self.bytes_to_torch(
                output,
                shape=(n_frames, height, width, 3),
                data_type=request.data_type,
                normalize=request.normalize,
            )
        return output


class _VideoFileManager(_MediaFileManager):
    managed_class = _VideoFile


class Video(_Media):
    _manager_class = _VideoFileManager
