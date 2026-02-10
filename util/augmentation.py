import numpy as np
from typing import Iterable


# Meta Augmentation Operations
class Augmentation:
    """Base class: ffmpeg filter graph builder"""

    def apply(self, stream, meta):
        return stream, meta

    def __call__(self, stream, meta):
        return self.apply(stream, meta)


class Compose(Augmentation):
    def __init__(self, ops: Iterable[Augmentation]):
        self.ops = ops

    def apply(self, stream, meta):
        for op in self.ops:
            stream, meta = op(stream, meta)
        return stream, meta


class RandomSelection(Augmentation):
    def __init__(self, ops: Iterable[Augmentation], n_apply=1, seed=None):
        self.ops = ops
        self.n_apply = n_apply
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        op = self.rng.choice(self.ops)
        for _ in range(self.n_apply - 1):
            op = self.rng.choice(self.ops)
            stream, meta = op(stream, meta)
        return stream, meta


class GaussianNoise(Augmentation):
    def __init__(self, max_noise=0.05, seed=None):
        self.rng = np.random.default_rng(seed)
        self.noise_level = max_noise

    def apply(self, stream, meta):
        w, h = meta.get("width"), meta.get("height")
        if not w or not h:
            return stream, meta
        stream = stream.filter(
            noise=f"alls={self.rng.uniform(0, self.noise_level)}")
        return stream, meta


# Vison Augmentation Operations
class RandomHFlip(Augmentation):
    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        w, h = meta.get("width"), meta.get("height")
        if not w or not h:
            return stream, meta
        if self.rng.random() < self.p:
            return stream.hflip(), meta
        return stream, meta


class RandomVFlip(Augmentation):
    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        w, h = meta.get("width"), meta.get("height")
        if not w or not h:
            return stream, meta
        if self.rng.random() < self.p:
            return stream.vflip(), meta
        return stream, meta


class RandomRotate(Augmentation):
    def __init__(self, max_deg=15.0, seed=None):
        self.max_deg = max_deg
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        w, h = meta.get("width"), meta.get("height")
        if not w or not h:
            return stream, meta
        angle = self.rng.uniform(-self.max_deg, self.max_deg) * np.pi / 180.0
        return stream.filter("rotate", angle), meta


class RandomCrop(Augmentation):
    def __init__(self, crop_ratio=0.8, seed=None):
        self.rng = np.random.default_rng(seed)
        self.crop_ratio = crop_ratio

    def apply(self, stream, meta):
        w, h = meta.get("width"), meta.get("height")
        if not w or not h:
            return stream, meta
            # Cannot crop if dimensions are unknown
            # This is assumed not to be image or video file
        cw, ch = int(w * self.crop_ratio), int(h * self.crop_ratio)

        x = self.rng.integers(0, w - cw + 1)
        y = self.rng.integers(0, h - ch + 1)
        return stream.crop(x, y, cw, ch).filter("scale", w, h), meta


class ColorJitter(Augmentation):
    def __init__(self, b=0.05, c=0.1, s=0.1, seed=None):
        self.rng = np.random.default_rng(seed)
        self.b = b
        self.c = c
        self.s = s

    def apply(self, stream, meta):
        w, h = meta.get("width"), meta.get("height")
        if not w or not h:
            return stream, meta
        bright = 1 + self.rng.uniform(-self.b, self.b)
        contrast = 1 + self.rng.uniform(-self.c, self.c)
        sat = 1 + self.rng.uniform(-self.s, self.s)

        return (
            stream.filter("eq", brightness=bright,
                          contrast=contrast, saturation=sat),
            meta,
        )


class Resize(Augmentation):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def apply(self, stream, meta):
        w, h = meta.get("width"), meta.get("height")
        if not w or not h:
            return stream, meta
        meta["width"] = self.width
        meta["height"] = self.height
        return stream.filter("scale", self.width, self.height), meta


# Video Specific Temporal Augmentations
class UniformTemporalSubFrame(Augmentation):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def apply(self, stream, meta):
        total_frames = meta.get("nb_frames", None)
        if not total_frames:
            return stream  # Cannot subsample if frame count is unknown
        step = max(total_frames // self.num_samples, 1)
        select_expr = "+".join(f"eq(n\\,{i})" for i in range(0,
                                                             total_frames, step)[:self.num_samples])
        meta["nb_frames"] = min(self.num_samples, total_frames)
        meta["frame_rate"] = meta["frame_rate"] *
        (meta["nb_frames"] / total_frames)
        return stream.filter("select", select_expr).filter("setpts", "N/(FRAME_RATE*TB)"), meta


# Audio Augmentations
class RandomAudioGain(Augmentation):
    def __init__(self, max_db=6.0, seed=None):
        self.max_db = max_db
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        if not meta.get("nb_channels", None):
            raise ValueError("File seems to have no audio stream.")
        gain = self.rng.uniform(-self.max_db, self.max_db)
        return stream.filter("volume", f"{gain}dB")


class RandomPitchShift(Augmentation):
    def __init__(self, semitone=2.0, seed=None):
        self.semitone = semitone
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        if not meta.get("nb_channels", None):
            raise ValueError("File seems to have no audio stream.")
        shift = self.rng.uniform(-self.semitone, self.semitone)
        ratio = 2 ** (shift / 12)  # semitone to freq ratio
        sr = meta["sample_rate"]
        meta["sample_rate"] = int(sr * ratio)
        meta["duration"] = meta["nb_samples"] / meta["sample_rate"]
        return stream.filter("asetrate", sr * ratio).filter("atempo", 1 / ratio), meta


class Resample(Augmentation):
    def __init__(self, target_sr):
        self.target_sr = target_sr

    def apply(self, stream, meta):
        if not meta.get("nb_channels", None):
            raise ValueError("File seems to have no audio stream.")
        meta["sample_rate"] = self.target_sr
        meta["nb_samples"] = int(meta["duration"] * self.target_sr)
        return stream.filter("aresample", self.target_sr), meta


class UniformAudioSubsample(Augmentation):
    def __init__(self, target_nb_samples):
        self.target_nb_samples = target_nb_samples

    def apply(self, stream, meta):
        if not meta.get("nb_samples", None):
            raise ValueError("File seems to have no audio stream.")
        total_samples = meta["nb_samples"]
        if total_samples <= self.target_nb_samples:
            return stream, meta  # No subsampling needed
        step = total_samples / self.target_nb_samples
        select_expr = "+".join(f"eq(n\\,{int(i * step)
                                         })" for i in range(self.target_nb_samples))
        meta["nb_samples"] = self.target_nb_samples
        meta["sample_rate"] = meta["sample_rate"] *
        (self.target_nb_samples / total_samples)
        return stream.filter("aselect", select_expr).filter("asetpts", "N/SR/TB"), meta
