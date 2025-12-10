import torch
import ffmpeg
import numpy as np
from typing import Union, Iterable
from ..system.threads import Thread
from .mediafile import (
    _Media,
    _MediaFile,
    _MediaFileManager,
    PathLike,
    MediaStreamRef,
    MediaDecodeRequest,
)


class _AudioFile(_MediaFile):
    def __init__(
        self,
        filename: PathLike,
    ):
        super().__init__(filename)
        self._metadata = Thread(target=self._initialize_streams, daemon=True)
        self._metadata.start()

    def _initialize_streams(self):
        self.streams = [
            _stream
            for _stream in ffmpeg.probe(self.filename)["streams"]
            if _stream["codec_type"] == "audio"
        ]
        self.formats = [_stream["codec_name"] for _stream in self.streams]
        self.channels = [int(_stream["channels"]) for _stream in self.streams]
        self.sample_rate = [int(_stream["sample_rate"]) for _stream in self.streams]
        self.duration = [float(_stream["duration"]) for _stream in self.streams]
        self.nb_samples = [int(_stream["nb_frames"]) for _stream in self.streams]
        self.tags = [_stream.get("tags", {}) for _stream in self.streams]

    def get_streams(self) -> Iterable[MediaStreamRef]:
        for idx, _ in enumerate(self.streams):
            yield MediaStreamRef(kind="audio", index=idx)

    def decode(
        self,
        request: "MediaDecodeRequest",
    ) -> Union[bytes, "np.ndarray", "torch.Tensor"]:
        # Implementation of audio decoding logic goes here
        if self._metadata is not None:
            self._metadata.join()
            self._metadata = None

        stream_idx = request.stream.index if request.stream else 0
        if stream_idx < 0 or stream_idx >= len(self.streams):
            raise IndexError("Audio stream index out of range")
        slice = request.time_slice
        if not isinstance(slice, slice):
            raise ValueError("Audio time_slice must be a slice object")
        start = slice.start or 0
        stop = slice.stop or self.duration[stream_idx]
        if start < 0 or stop > self.duration[stream_idx] or start >= stop:
            raise IndexError("Time slice out of range")
        start_time = start
        duration = stop - start

        ffmpeg_stream = ffmpeg.input(self.filename, ss=start_time, t=duration)

        augments = request.augments if request.stream else None
        meta_data = {
            "index": stream_idx,
            "channels": self.channels[stream_idx],
            "sample_rate": self.sample_rate[stream_idx],
            "duration": duration,
            "nb_samples": int(self.sample_rate[stream_idx] * duration),
        }
        if request.augments:
            ffmpeg_stream, meta_data = augments.apply(
                ffmpeg_stream,
                meta=meta_data,
            )
        out, _ = ffmpeg_stream.output(
            "pipe:",
            format="wav",
            acodec="pcm_s16le",
            ac=self.channels[stream_idx],
            ar=self.sample_rate[stream_idx],
        ).run(capture_stdout=True, capture_stderr=True)
        n_ch = meta_data["channels"]
        n_sp = meta_data["nb_samples"]
        d_ty = request.data_type
        if request.return_type == "numpy":
            return self.bytes_to_numpy(
                out, (n_ch, n_sp), dtype=d_ty, normalize=request.normalize
            )
        elif request.return_type == "torch":
            return self.bytes_to_torch(
                out, (n_ch, n_sp), dtype=d_ty, normalize=request.normalize
            )
        else:
            return out


class _AudioFileManager(_MediaFileManager):
    _managed_class = _AudioFile


class Audio(_Media):
    _manager_class = _AudioFileManager
