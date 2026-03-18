from .augmentation import VideoAugmentation


class UniformTemporalSubFrame(VideoAugmentation):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def apply(self, stream, meta):
        total_frames = meta.get("nb_frames", None)
        if not total_frames:
            return stream  # Cannot subsample if frame count is unknown
        step = max(total_frames // self.num_samples, 1)
        select_expr = "+".join(
            f"eq(n,{i})"
            for i in range(0, total_frames, step)[:self.num_samples]
        )
        meta["nb_frames"] = min(self.num_samples, total_frames)
        meta["frame_rate"] = meta["frame_rate"] * \
            (meta["nb_frames"] / total_frames)
        return (
            stream
            .filter("select", select_expr)
            .filter("setpts", "N/(FRAME_RATE*TB)")
        ), meta
