import numpy as np
from .augmentation import AudioAugmentation


class RandomPitch(AudioAugmentation):
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
        return (
            stream
            .filter("asetrate", sr * ratio)
            .filter("atempo", 1 / ratio)
        ), meta
