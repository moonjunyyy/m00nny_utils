import numpy as np
from .augmentation import VideoAugmentation


class RandomTemporalCrop(VideoAugmentation):
    """A clip of num_frames consecutive frames from a random start; a
    shorter video is padded with its last frame."""

    def __init__(self, num_frames, seed=None):
        self.num_frames = num_frames
        self.rng = np.random.default_rng(seed)

    def apply(self, stream, meta):
        total_frames = meta.get("nb_frames", None)
        if not total_frames:
            return stream, meta  # Cannot crop if frame count is unknown
        if total_frames > self.num_frames:
            start = int(self.rng.integers(
                0, total_frames - self.num_frames + 1))
            stream = stream.trim(
                start_frame=start, end_frame=start + self.num_frames
            ).setpts("PTS-STARTPTS")
        elif total_frames < self.num_frames:
            stream = stream.filter(
                "tpad", stop_mode="clone", stop=self.num_frames - total_frames)
        meta["nb_frames"] = self.num_frames
        if meta.get("frame_rate"):
            meta["duration"] = self.num_frames / meta["frame_rate"]
        return stream, meta
