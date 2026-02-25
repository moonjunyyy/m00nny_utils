import os
import torch
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Optional, Tuple, TypeVar

PathLike = TypeVar("PathLike", str, bytes, os.PathLike)
ImageLike = TypeVar("ImageLike", Image.Image, torch.Tensor, np.ndarray)


class UCF101(Dataset):
    def __init__(
        self,
        root: PathLike,
        train: bool = True,
        transform: Callable[[ImageLike], torch.Tensor] | None = None,
        target_transform: Callable[[ImageLike], torch.Tensor] | None = None,
        download: bool = False,
    ) -> None:
        root = os.path.join(root, "ucf101/videos")
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.frames_per_clip = 16

        # List all video paths and labels
        self.video_paths = []
        self.targets = []
        self.class_to_idx = {}

        self.classes = [f"{i}" for i in range(101)]

        for idx, class_name in enumerate(sorted(os.listdir(root))):
            class_dir = os.path.join(root, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                for video_file in os.listdir(class_dir):
                    if video_file.endswith(".avi"):
                        self.video_paths.append(os.path.join(class_dir, video_file))
                        self.targets.append(idx)

        self.domains = [0 for _ in range(len(self.video_paths))]
        self.domain_names = ["domain_0" for _ in range(len(self.video_paths))]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.targets[idx]

        frames = self.load_video(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        video_tensor = torch.stack(frames)  # (T, C, H, W)
        return video_tensor, label, self.domains[idx]

    # def load_video(self, path):
    #     cap = cv2.VideoCapture(path)
    #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #     frame_indices = self.sample_frame_indices(total_frames)
    #     frames = []

    #     for i in frame_indices:
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    #         ret, frame = cap.read()
    #         if ret:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             frames.append(transforms.ToPILImage()(frame))
    #         else:
    #             break
    #     cap.release()
    #     return frames
    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = self.sample_frame_indices(total_frames)
        frames = []

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(transforms.ToPILImage()(frame))
            else:
                # If failed to read frame, pad with the last valid frame
                if frames:
                    frames.append(frames[-1])
                else:
                    # If first frame fails, pad with a black frame
                    frames.append(Image.new("RGB", (224, 224), (0, 0, 0)))

        cap.release()

        # If still not enough frames, pad at the end
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])

        # If too many (shouldn't happen), trim
        if len(frames) > self.frames_per_clip:
            frames = frames[: self.frames_per_clip]

        return frames

    def sample_frame_indices(self, total_frames):
        if total_frames < self.frames_per_clip:
            return list(range(total_frames)) + [total_frames - 1] * (
                self.frames_per_clip - total_frames
            )
        start = torch.randint(0, total_frames - self.frames_per_clip + 1, (1,)).item()
        return list(range(start, start + self.frames_per_clip))
