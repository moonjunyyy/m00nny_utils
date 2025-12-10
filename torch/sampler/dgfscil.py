import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from typing import Optional, Sized, Iterable, Tuple, Union
from ..datasets.DomainDataset import DomainDataset

class DGFSCILSampler:
    def __init__(
            self,
            data_source: Sized,
            num_tasks: int,
            rnd_seed: int,
            cur_iter: int= 0,
            num_replicas: int=None,
            rank: int=None,
            *,
            task_config: Optional[Iterable[Iterable[Tuple[int, int]]]] = None,
    ) -> None:
        """
        Sampler for Distributed Data Parallel training on Domain Generalization
        datasets like CORe50, PACS, etc.
        No more inherits from torch.utils.data.Sampler,
        since it does not support indexing.

        Args:
            data_source (Sized): Dataset to sample from.
            num_tasks (int): Number of tasks (domains).
            rnd_seed (int): Random seed for shuffling.
            cur_iter (int, optional): Current iteration for varying N and M. Defaults to 0.
            num_replicas (int, optional): Number of processes in distributed training.
                If None, will be retrieved from torch.distributed. Defaults to None.
            rank (int, optional): Rank of the current process in distributed training.
                If None, will be retrieved from torch.distributed. Defaults to None.

        Kwargs:
            task_config (Optional[Iterable[Iterable[Tuple[int, int]]]], optional):
                Configuration of tasks specifying (Domain, Class) pairs for each task.
                If the length not equal to num_tasks, an error is raised.
                This strictly defines the sampling behavior, may remove some categories.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_tasks = num_tasks
        self.rnd_seed = rnd_seed
        self.cur_iter = cur_iter
        self.task_config = task_config
        if self.task_config is not None:
            if len(self.task_config) != self.num_tasks:
                raise ValueError("Length of task_config must be equal to num_tasks")
        self.tasks = [[] for _ in range(self.num_tasks)]
        self._task_ready = False
        self._bootstrap()
        
    def _config_tasks(self, style: str = "CIL"):
        # Default CIL configuration
        self.domain_class_indices = []
        domains = set([d for d in self.data_source.domain_names])
        classes = set([c for c in self.data_source.class_names])
        n_domain = len(domains)
        n_class = len(classes)
        classes_per_task = (
            num_classes // self.num_tasks
            + (1 if num_classes % self.num_tasks > 0 else 0)
        )
        task_config = []
        class_list = list(classes)
        domain_vec = torch.arange(n_domain).view(-1, 1, 1).repeat(1, n_class, 1)
        class_vec = torch.arange(n_class).view(1, -1, 1).repeat(n_domain, 1, 1)
        domain_class_pairs = torch.cat([domain_vec, class_vec], dim=-1)

        if style = "CIL":
            class_permute = torch.randperm(n_class, generator=torch.Generator().manual_seed(self.rnd_seed))
            for task_id in range(self.num_tasks):
                selected_classes = class_permute[
                    task_id * classes_per_task : (task_id + 1) * classes_per_task
                ]
                task_config.append(
                    domain_class_pairs[:, selected_classes, :].view(-1, 2).tolist()
                )
        elif style == "DIL":
            domain_permute = torch.randperm(n_domain, generator=torch.Generator().manual_seed(self.rnd_seed))
            for task_id in range(self.num_tasks):
                selected_domain = domain_permute[task_id % n_domain]
                selected_classes = torch.randperm(n_class, generator=torch.Generator().manual_seed(self.rnd_seed + task_id))[:classes_per_task]
                task_config.append(
                    domain_class_pairs[selected_domain, selected_classes, :].view(-1, 2).tolist()
                )
        elif style == "VIL":
            cross_permute = torch.randperm(n_domain * n_class, generator=torch.Generator().manual_seed(self.rnd_seed))
            for task_id in range(self.num_tasks):
                selected_pairs = cross_permute[
                    task_id * classes_per_task : (task_id + 1) * classes_per_task
                ]
                selected_domain = selected_pairs // n_class
                selected_classes = selected_pairs % n_class
                domain_class_pairs = torch.stack([selected_domain, selected_classes], dim=-1)
                task_config.append(
                    domain_class_pairs.view(-1, 2).tolist()
                )
        self.task_config = task_config


    def _bootstrap(self):
            # Strictly follow the task_config
            self.domain_class_indices = []
            for task_id in range(self.num_tasks):
                domain_class_pairs = self.task_config[task_id]
                indices = []
                for domain_id, class_id in domain_class_pairs:
                    indices += [
                        idx for idx, (d, c) in enumerate(self.data_source.domains_targets)
                        if d == domain_id and c == class_id
                    ]
                self.domain_class_indices.append(indices)



