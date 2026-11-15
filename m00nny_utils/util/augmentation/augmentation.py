# Meta Class for Augmentation
class Augmentation:
    """Base class: ffmpeg filter graph builder.

    An operation takes the stream being built and the metadata of what is
    being decoded and returns both: apply(stream, meta) -> (stream, meta).
    `requires` names the metadata it cannot do without; calling the
    operation checks that before apply.
    """
    requires = ()

    def __init__(self, *args, **kwargs):
        pass

    def apply(self, stream, meta):
        return stream, meta

    def _ensure_in_meta(self, meta, keys):
        for key in keys:
            assert key in meta, \
                f"{key} is required in meta for {self.__class__.__name__}"

    def __call__(self, stream, meta):
        self._ensure_in_meta(meta, self.requires)
        return self.apply(stream, meta)


# Meta Class for Vision Specific Augmentation
class VisionAugmentation(Augmentation):
    requires = ("width", "height")


# Meta Class for Audio Specific Augmentation
class AudioAugmentation(Augmentation):
    requires = ("sample_rate",)


# Meta Class for Video Specific Augmentation
class VideoAugmentation(VisionAugmentation):
    requires = ("width", "height", "frame_rate")
