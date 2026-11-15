import numpy as np
import ffmpeg
from .augmentation import Augmentation


class AddWhiteNoise(Augmentation):
    def __init__(self, max_noise=0.05, seed=None):
        self.rng = np.random.default_rng(seed)
        self.max_noise = max_noise

    def apply(self, stream, meta):
        # Check the data is whether audio or video
        if meta.get("nb_channels", None):  # Audio stream
            noise_level = self.rng.uniform(0, self.max_noise)
            return (
                ffmpeg
                .concat(
                    stream
                    .filter(
                        "anoisesrc",
                        d=meta["duration"],
                        r=meta["sample_rate"],
                        a=self.max_noise,
                        all_seed=self.rng.integers(0,  2 << 31)
                    ),
                    stream,
                )
                .filter("amix", inputs=2, duration="first"),
            ), meta
        elif meta.get("width", None) and meta.get("height", None):  # Video stream
            noise_level = self.rng.uniform(0, self.max_noise)
            return (
                stream
                .filter(
                    "noise",
                    all_seed=self.rng.integers(0, 2 << 31),
                    all_strength=noise_level,
                    d=meta["duration"],
                ),
            ), meta
        else:
            return stream, meta  # Cannot add noise if type is unknown
