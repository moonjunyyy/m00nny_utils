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
        # no end given: to the end of what is being decoded, each time
        end_time = meta["duration"] if self.end_time is None else self.end_time
        meta["duration"] = end_time - self.start_time
        meta["nb_samples"] = int(meta["duration"] * meta["sample_rate"])
        return (
            stream
            .filter(
                "atrim",
                start=self.start_time,
                end=end_time
            )
            .filter("asetpts", "PTS-STARTPTS")
        ), meta


class TrimVideo(VideoAugmentation):
    def __init__(self, start_time=0.0, end_time=None):
        self.start_time = start_time
        self.end_time = end_time

    def apply(self, stream, meta):
        if not meta.get("nb_frames", None):
            raise ValueError("File seems to have no video stream.")
        end_time = meta["duration"] if self.end_time is None else self.end_time
        meta["duration"] = end_time - self.start_time
        meta["nb_frames"] = int(meta["duration"] * meta["frame_rate"])
        return (
            stream
            .filter(
                "trim",
                start=self.start_time,
                end=end_time
            )
            .filter("setpts", "PTS-STARTPTS")
        ), meta
