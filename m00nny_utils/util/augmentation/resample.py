from .augmentation import AudioAugmentation


class Resample(AudioAugmentation):
    def __init__(self, target_sr):
        self.target_sr = target_sr

    def apply(self, stream, meta):
        if not meta.get("nb_channels", None):
            raise ValueError("File seems to have no audio stream.")
        meta["sample_rate"] = self.target_sr
        meta["nb_samples"] = int(meta["duration"] * self.target_sr)
        return stream.filter("aresample", self.target_sr), meta
