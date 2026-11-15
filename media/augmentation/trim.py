from .augmentation import AudioAugmentation, VideoAugmentation

# Temporal Trimming for Alignment
# Crop -> Spatial
# Trim -> Temporal


class TrimAudio(AudioAugmentation):
    def __init__(self, start_time=0.0, end_time=None):
        self.start_time = start_time
        self.end_time = end_time

    def apply(self, stream, meta):
        if not meta.get("nb_channels", None):
            raise ValueError("File seems to have no audio stream.")
        if self.end_time is None:
            self.end_time = meta["duration"]
        meta["duration"] = self.end_time - self.start_time
        meta["nb_samples"] = int(meta["duration"] * meta["sample_rate"])
        return stream.filter(
            "atrim",
            start=self.start_time,
            end=self.end_time
        ), meta


class TrimVideo(VideoAugmentation):
    def __init__(self, start_time=0.0, end_time=None):
        self.start_time = start_time
        self.end_time = end_time

    def apply(self, stream, meta):
        if not meta.get("nb_frames", None):
            raise ValueError("File seems to have no video stream.")
        if self.end_time is None:
            self.end_time = meta["duration"]
        meta["duration"] = self.end_time - self.start_time
        meta["nb_frames"] = int(meta["duration"] * meta["frame_rate"])
        return (
            stream
            .filter(
                "trim",
                start=self.start_time,
                end=self.end_time
            )
            .filter("setpts", "PTS-STARTPTS")
        ), meta
