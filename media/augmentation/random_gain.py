import numpy as np
from .augmentation import AudioAugmentation


# Audio Augmentations
class RandomGain(AudioAugmentation):
    def __init__(self, max_db=6.0, seed=None):
        self.max_db = max_db
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        if not meta.get("nb_channels", None):
            raise ValueError("File seems to have no audio stream.")
        gain = self.rng.uniform(-self.max_db, self.max_db)
        return stream.filter("volume", f"{gain}dB")
