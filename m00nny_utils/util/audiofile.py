import torch
import ffmpeg
import numpy as np
from typing import Union, Iterable
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


class _AudioFile(_MediaFile):
    _kind = "audio"

    def __init__(
        self,
        filename: PathLike,
    ):
        super().__init__(filename)
        self._metadata = Task(target=self._initialize_streams, daemon=True)
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
        # Containers that do not store the sample count (wav, flac): duration x sample rate.
        self.nb_samples = [
            int(_stream["nb_frames"]) if "nb_frames" in _stream
            else int(float(_stream["duration"]) * int(_stream["sample_rate"]))
            for _stream in self.streams
        ]
        self.tags = [_stream.get("tags", {}) for _stream in self.streams]

    def decode(
        self,
        request: "MediaDecodeRequest",
    ) -> Union[bytes, "np.ndarray", "torch.Tensor"]:
        # Implementation of audio decoding logic goes here
        self._ready()

        stream_idx = request.stream.index if request.stream else 0
        if stream_idx < 0 or stream_idx >= len(self.streams):
            raise IndexError("Audio stream index out of range")
        time_slice = request.time_slice
        if time_slice is None:
            time_slice = slice(None)  # the whole stream
        if not isinstance(time_slice, slice):
            raise ValueError("Audio time_slice must be a slice object")
        start = time_slice.start or 0
        stop = time_slice.stop or self.duration[stream_idx]
        if start < 0 or stop > self.duration[stream_idx] or start >= stop:
            raise IndexError("Time slice out of range")
        start_time = start
        duration = stop - start

        ffmpeg_stream = ffmpeg.input(self.filename, ss=start_time, t=duration)

        augments = request.aug_policy
        meta_data = {
            "index": stream_idx,
            "nb_channels": self.channels[stream_idx],
            "sample_rate": self.sample_rate[stream_idx],
            "duration": duration,
            "nb_samples": int(self.sample_rate[stream_idx] * duration),
        }
        if augments:
            ffmpeg_stream, meta_data = augments.apply(
                ffmpeg_stream,
                meta=meta_data,
            )
        # Raw interleaved 16-bit samples (no container header), at the channel count and rate the metadata says
        # after the augmentations.
        n_ch = meta_data["nb_channels"]
        out = self._run(ffmpeg_stream.output(
            "pipe:",
            format="s16le",
            acodec="pcm_s16le",
            ac=n_ch,
            ar=meta_data["sample_rate"],
        ))
        # The sample count is what the decoder produced (the metadata's count is an estimate at slice borders).
        n_sp = len(out) // (2 * n_ch)
        d_ty = request.data_type
        # The decoder interleaves the channels, (samples, channels); returned as (channels, samples).
        if request.return_type == MediaFileReturnType.NumPy:
            return self.bytes_to_ndarray(
                out, (n_sp, n_ch), data_type=d_ty, original_dtype=MediaFileDataType.Int16, normalize=request.normalize
            ).T
        elif request.return_type == MediaFileReturnType.PyTorch:
            return self.bytes_to_torch(
                out, (n_sp, n_ch), data_type=d_ty, original_dtype=MediaFileDataType.Int16, normalize=request.normalize
            ).T
        else:
            return out


class _AudioFileManager(_MediaFileManager):
    _managed_class = _AudioFile


class Audio(_Media):
    _manager_class = _AudioFileManager
