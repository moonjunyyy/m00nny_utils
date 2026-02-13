import torch


class CILSampler:
    def __init__(
        self,
        data_source: torch.utils.data.Dataset,
        num_tasks: int,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
    ):
        self.data_source = data_source
        self.num_tasks = num_tasks
        self.shuffle = shuffle
        self.rnd_seed = seed
        self.task_id = None

        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self._bootstrap()

    def _bootstrap(self):
        # Create a simple class-incremental learning sampler
        num_classes = len(set(self.data_source.targets))
        classes_per_task = (
            num_classes // self.num_tasks
            + (1 if num_classes % self.num_tasks > 0 else 0)
        )
        class_permute = torch.randperm(
            num_classes,
            generator=torch.Generator().manual_seed(self.rnd_seed)
        )
        self.task_config = []
        for task_id in range(self.num_tasks):
            selected_classes = class_permute[
                task_id * classes_per_task: (task_id + 1) * classes_per_task
            ]
            self.task_config.append(selected_classes.tolist())

    def set_task(self, task_id: int):
        # Set the current task and create indices for the sampler
        selected_classes = self.task_config[task_id]
        self.task_id = task_id
        self.indices = [
            idx for idx, target in enumerate(self.data_source.targets)
            if target in selected_classes
        ]
        len_data = len(self.indices) // self.num_replicas
        if not self.drop_last and len(self.indices) % self.num_replicas != 0:
            len_data += 1
        self.indices = self.indices[
            self.rank: len_data * self.num_replicas: self.num_replicas
        ]

    def get_task(self, task_id=None):
        # Get the indices for a specific task
        if task_id is None:
            task_id = self.task_id
        selected_classes = self.task_config[task_id]
        return selected_classes

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(
                len(self.indices),
                generator=torch.Generator().manual_seed(
                    self.rnd_seed + self.task_id
                    if self.task_id is not None
                    else self.rnd_seed
                ))
            shuffled_indices = [self.indices[i] for i in perm]
            return iter(shuffled_indices)
        else:
            return iter(self.indices)

    def __len__(self):
        return len(self.indices)
