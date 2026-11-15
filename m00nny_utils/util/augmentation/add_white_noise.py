import numpy as np
import ffmpeg
from .augmentation import Augmentation


class AddWhiteNoise(Augmentation):
    """White noise of a random level up to max_noise, a fraction of full
    scale, on sound or on pictures."""

    def __init__(self, max_noise=0.05, seed=None):
        self.rng = np.random.default_rng(seed)
        self.max_noise = max_noise

    def apply(self, stream, meta):
        # Check the data is whether audio or video
        if meta.get("nb_channels", None):  # Audio stream
            noise_level = self.rng.uniform(0, self.max_noise)
            # anoisesrc is a source of its own, mixed into the sound;
            # normalize=0: the sound keeps its level
            noise = ffmpeg.input(
                f"anoisesrc=d={meta['duration']}:r={meta['sample_rate']}"
                f":a={noise_level}:seed={int(self.rng.integers(0, 1 << 31))}",
                format="lavfi",
            )
            return ffmpeg.filter(
                [stream, noise],
                "amix",
                inputs=2,
                duration="first",
                normalize=0,
            ), meta
        elif meta.get("width", None) and meta.get("height", None):  # Video stream
            noise_level = self.rng.uniform(0, self.max_noise)
            return stream.filter(
                "noise",
                all_seed=int(self.rng.integers(0, 1 << 31)),
                # the filter counts strength from 0 to 100
                all_strength=int(round(noise_level * 100)),
            ), meta
        else:
            return stream, meta  # Cannot add noise if type is unknown
