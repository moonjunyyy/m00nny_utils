import torch
from itertools import product
from torch.utils.data.sampler import Sampler
from typing import Callable, Hashable, Iterable, List, Optional, Sequence, Sized, Tuple, Union
from ...util.labels import UNLABELED
from .sampler_base import _ReplicaSampler

# What the tasks are split by: a label axis of the dataset (the name of a per-sample attribute: "targets",
# "domains", "super_classes", ...), several axes together (their combinations are the categories), or a function
# giving the key of every sample.
#   "targets"                class-incremental          (cil)
#   "domains"                domain-incremental         (dil)
#   ("domains", "targets")   every (domain, class) pair (vil)
Criterion = Union[str, Sequence[str], Callable[[Sized], Sequence[Hashable]]]



def _pretty_print_nested_lists_in_double_columns(
        left,
        right,
        left_title=None,
        right_title=None,
        sep=" ",
        max_len=96
):
    if left_title is None:
        left_title = ""
    if right_title is None:
        right_title = ""
    left_strs = [str(x) for x in left]
    right_strs = [str(x) for x in right]
    left_len = min(
        max([len(x) for x in left_strs]) + 1,
        max_len
    )
    right_len = min(
        max([len(x) for x in right_strs]) + 1,
        max_len
    )
    if len(left_strs) < len(right_strs):
        left_strs += [""] * (len(right_strs) - len(left_strs))
    if len(left_strs) > len(right_strs):
        right_strs += [""] * (len(left_strs) - len(right_strs))
    for _rank, (l, r) in enumerate(zip(left_strs, right_strs)):
        # if longer than left_len (max 24), slice it into 24 in multiple lines
        # if len(l) > left_len or len(r) > right_len:
        _split_l = l.split(" ")
        _split_r = r.split(" ")
        # Devide into multiple lines
        _l_buffer = [""]
        for _lw in _split_l:
            if len(_l_buffer) == 0 and len(_l_buffer[0]) == 0:
                _l_buffer.append(_lw)
            elif len(_l_buffer[-1]) + len(_lw) + 1 > left_len:
                _l_buffer.append("   " + _lw)
            else:
                _l_buffer[-1] += " " + _lw

        _l_max = max([len(x) for x in _l_buffer])
        if _l_max > left_len:
            left_len = _l_max
        _r_buffer = [""]
        for _rw in _split_r:
            if len(_r_buffer) == 0 and len(_r_buffer[0]) == 0:
                _r_buffer.append(_rw)
            elif len(_r_buffer[-1]) + len(_rw) + 1 > right_len:
                _r_buffer.append("   " + _rw)
            else:
                _r_buffer[-1] += " " + _rw
        _r_max = max([len(x) for x in _r_buffer])
        if _r_max > right_len:
            right_len = _r_max
        if len(_r_buffer) > len(_l_buffer):
            _l_buffer += [""] * (len(_r_buffer) - len(_l_buffer))
        if len(_r_buffer) < len(_l_buffer):
            _r_buffer += [""] * (len(_l_buffer) - len(_r_buffer))
        left_strs[_rank] = _l_buffer
        right_strs[_rank] = _r_buffer
    print(f"\n{left_title:{left_len}}{sep}{right_title}")
    for _rank, (l, r) in enumerate(zip(left_strs, right_strs)):
        print(f"{'':{left_len}}{sep}{'':{right_len}}")
        if isinstance(l, list) or isinstance(r, list):
            for _l, _r in zip(l, r):
                print(f"{_l:{left_len}}{sep}{_r}")
        else:
            print(f"{l:{left_len}}{sep}{r}")
    print()


def _axis(data_source: Sized, name: str) -> List[Hashable]:
    """The value of one label axis for every sample."""
    if not hasattr(data_source, name):
        raise AttributeError(
            f"{type(data_source).__name__} has no label axis {name!r} (a per-sample attribute such as targets)"
        )
    values = getattr(data_source, name)
    values = values.tolist() if torch.is_tensor(values) else [v.item() if torch.is_tensor(v) else v for v in values]
    if len(values) != len(data_source):
        raise ValueError(f"label axis {name!r} has {len(values)} values for {len(data_source)} samples")
    return values


def label_keys(data_source: Sized, criterion: Criterion) -> Tuple[List[Hashable], List[Hashable]]:
    """
    The key of every sample under a criterion, and the categories: the keys the tasks are split over, in a fixed
    order. For several axes a key is the tuple of their values and the categories are all combinations (the first
    axis is the major one). A value UNLABELED on an axis is not a category.
    """
    if callable(criterion):
        keys = list(criterion(data_source))
        if len(keys) != len(data_source):
            raise ValueError(f"the criterion gave {len(keys)} keys for {len(data_source)} samples")
        return keys, sorted(set(k for k in keys if k != UNLABELED))
    if isinstance(criterion, str):
        keys = _axis(data_source, criterion)
        return keys, sorted(set(k for k in keys if k != UNLABELED))
    axes = [_axis(data_source, name) for name in criterion]
    keys = list(zip(*axes))
    categories = list(product(*[sorted(set(v for v in axis if v != UNLABELED)) for axis in axes]))
    return keys, categories


class ContinualSampler(_ReplicaSampler):
    def __init__(
            self,
            data_source: Sized,
            criterion: Criterion = "targets",
            num_tasks: int = 1,
            m: int = 0,
            n: int = 100,
            rnd_seed: Optional[int] = None,
            cur_iter: int = 0,
            varing_NM: bool = False,
            batchsize: Optional[int] = None,
            online_iter: int = 1,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            *,
            task_config: Optional[Iterable[Iterable[Hashable]]] = None,
            verbose: bool = True,
    ) -> None:
        """
        The samples of a dataset split into tasks, one task at a time (set_task).

        The split is made over categories: the values of `criterion` (see Criterion). N% of the categories are
        disjoint: each belongs to one task. The others are blurry: each still has a home task, but M% of their
        samples are shuffled across the tasks. With one task (num_tasks=1) this is joint training.

        Args:
            data_source: the dataset; it has the label axes the criterion names, one value per sample.
            criterion: what the tasks are split by.
            num_tasks: number of tasks.
            m: percentage of the blurry categories' samples that are shuffled across tasks.
            n: percentage of the categories that are disjoint.
            rnd_seed: seed of the split and of the order within a task (None: not seeded).
            cur_iter: the task to start from.
            varing_NM: tasks get unequal numbers of categories and of shuffled samples (random cut points) instead
                of equal shares.
            batchsize, online_iter: with a batch size, the order of a task is cut into batches and each batch is
                given online_iter times in a row (online training: several iterations on the same stream batch).
            num_replicas, rank: distributed training: this process (rank) takes every num_replicas-th sample.

        Kwargs:
            task_config: the categories of each task, given explicitly instead of the N/M split: one list of
                category values per task, e.g. [[(domain, class), ...], ...] for ("domains", "targets"). Categories
                that are not listed are in no task.
            verbose: print the split.
        """
        super().__init__()
        self.data_source = data_source
        self.criterion   = criterion
        self.num_tasks   = num_tasks
        self.m           = m
        self.n           = n
        self.rnd_seed    = rnd_seed
        self.varing_NM   = varing_NM
        self.batchsize   = batchsize
        self.online_iter = online_iter
        self.generator   = torch.Generator().manual_seed(rnd_seed) if rnd_seed is not None else torch.Generator()
        self._init_replicas(num_replicas, rank)

        self.keys, self.categories = label_keys(data_source, criterion)
        category_index = {category: i for i, category in enumerate(self.categories)}

        if task_config is not None:
            task_config = [list(task) for task in task_config]
            if len(task_config) != num_tasks:
                raise ValueError("Length of task_config must be equal to num_tasks")
            unknown = [c for task in task_config for c in task if c not in category_index]
            if unknown:
                raise ValueError(f"task_config names categories the dataset does not have: {unknown[:8]}")
            self.disjoint_classes = [[category_index[c] for c in task] for task in task_config]
            self.blurry_classes   = [[] for _ in range(num_tasks)]
        else:
            self.disjoint_classes, self.blurry_classes = self._split_disjoint_blurry_categories(
                len(self.categories), num_tasks, n, varing_NM
            )
        if verbose:
            _pretty_print_nested_lists_in_double_columns(
                [[self.categories[c] for c in task] for task in self.disjoint_classes],
                [[self.categories[c] for c in task] for task in self.blurry_classes],
                "disjoint categories", "blurry categories", sep=" | ")

        # Get indices of disjoint and blurry categories
        self.disjoint_indices = [[] for _ in range(num_tasks)]
        self.blurry_indices   = [[] for _ in range(num_tasks)]
        home = {}   # category -> the index list of its task
        for task in range(num_tasks):
            for c in self.disjoint_classes[task]: home[c] = self.disjoint_indices[task]
            for c in self.blurry_classes[task]:   home[c] = self.blurry_indices[task]
        for i, key in enumerate(self.keys):
            indices = home.get(category_index.get(key))
            if indices is not None:
                indices.append(i)

        # Randomly shuffle M% of blurry indices
        if not varing_NM:
            num_blurreds = [len(self.blurry_indices[i]) * m // 100 for i in range(num_tasks)]
        else:
            num_blurred = sum(len(indices) for indices in self.blurry_indices) * m // 100
            num_blurreds = self._cut_sizes(num_blurred, num_tasks)
        blurred = []
        for i in range(num_tasks):
            taken = self.blurry_indices[i][:num_blurreds[i]]
            num_blurreds[i] = len(taken)   # a task gets back as many as it gave
            blurred += taken
            self.blurry_indices[i] = self.blurry_indices[i][len(taken):]
        blurred = torch.tensor(blurred, dtype=torch.long)
        blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
        if verbose:
            print("blurry indices: ", len(blurred))
        for i in range(num_tasks):
            self.blurry_indices[i] += blurred[:num_blurreds[i]]
            blurred = blurred[num_blurreds[i]:]

        self.indices = [[] for _ in range(num_tasks)]
        for i in range(num_tasks):
            if verbose:
                print("task %d: disjoint %d, blurry %d, online_iter : %d" % (
                    i, len(self.disjoint_indices[i]), len(self.blurry_indices[i]), self.online_iter))
            indices = torch.tensor(self.disjoint_indices[i] + self.blurry_indices[i], dtype=torch.long)
            indices = indices[torch.randperm(len(indices), generator=self.generator)]
            if batchsize is not None:
                # Each batch online_iter times in a row; the last, shorter batch too.
                num_batches = int(indices.size(0) // batchsize)
                full, rest  = indices[:num_batches * batchsize], indices[num_batches * batchsize:]
                indices     = torch.cat([
                    full.reshape(-1, int(batchsize)).repeat(1, int(online_iter)).flatten(),
                    rest.repeat(int(online_iter)),
                ])
            self.indices[i] = indices.tolist()

        self.set_task(cur_iter)

    def _cut_sizes(self, total: int, num_tasks: int) -> List[int]:
        """total cut into num_tasks consecutive parts at random points (parts may be empty): their sizes."""
        if total == 0:
            return [0] * num_tasks
        cuts = [0] + torch.randint(0, total, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [total]
        return [cuts[i + 1] - cuts[i] for i in range(num_tasks)]

    def _cut(self, values: List, num_tasks: int) -> List[List]:
        sizes, parts = self._cut_sizes(len(values), num_tasks), []
        for size in sizes:
            parts.append(values[:size])
            values = values[size:]
        return parts

    def _split_disjoint_blurry_categories(self,
                                          num_categories                  :int,
                                          num_tasks                       :int,
                                          n                               :int,
                                          varing_NM                       :bool=False):
        # Divide categories into N% of disjoint and (100 - N)% of blurry; each kind is shared out to the tasks
        # (equally, the categories that do not divide are left out; or at random cut points with varing_NM).
        disjoint_num   = num_categories * n // 100
        disjoint_num   = int(disjoint_num // num_tasks) * num_tasks
        blurry_num     = num_categories - disjoint_num
        blurry_num     = int(blurry_num   // num_tasks) * num_tasks

        category_order   = torch.randperm(num_categories, generator=self.generator)
        disjoint_classes = category_order[:disjoint_num]
        blurry_classes   = category_order[disjoint_num:disjoint_num + blurry_num]
        if not varing_NM:
            disjoint_classes = disjoint_classes.reshape(num_tasks, disjoint_num // num_tasks).tolist()
            blurry_classes   = blurry_classes.reshape(num_tasks, blurry_num // num_tasks).tolist()
        else:
            disjoint_classes = self._cut(disjoint_classes.tolist(), num_tasks)
            blurry_classes   = self._cut(blurry_classes.tolist(), num_tasks)
        return disjoint_classes, blurry_classes

    def __iter__(self) -> Iterable[int]:
        return iter(self.get_task(self.task))

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int) -> None:
        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter
        self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = self.num_samples

    def get_task(self, cur_iter: int) -> List[int]:
        """The indices of a task (this replica's share)."""
        return self._share(self.indices[cur_iter])

    def get_task_categories(self, cur_iter: int) -> List[Hashable]:
        """The categories that appear in a task (its own and the ones blurred into it)."""
        return sorted(set(self.keys[i] for i in self.indices[cur_iter]))

    def get_task_labels(self, cur_iter: int, axis: str) -> List[Hashable]:
        """The values of a label axis that appear in a task."""
        values = _axis(self.data_source, axis)
        return sorted(set(values[i] for i in self.indices[cur_iter]))

    def get_task_classes(self, cur_iter: int) -> List[Hashable]:
        return self.get_task_labels(cur_iter, "targets")


class ContinualTestSampler(_ReplicaSampler):
    def __init__(
            self,
            data_source: Sized,
            criterion: Criterion = "targets",
            exposed: Iterable[Hashable] = (),
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
    ) -> None:
        """
        The samples whose category has been exposed (seen in training so far).

        Args:
            data_source: the dataset.
            criterion: the same criterion as the training sampler's.
            exposed: the exposed categories (ContinualSampler.get_task_categories of the tasks so far).
            num_replicas, rank: distributed evaluation.

        A sample that is UNLABELED on some axes (e.g. a test set without domains) is selected when the axes it
        does have match an exposed category.
        """
        super().__init__()
        self.data_source = data_source
        self.criterion   = criterion
        self.exposed     = list(exposed)
        self._init_replicas(num_replicas, rank)

        self.keys, _ = label_keys(data_source, criterion)
        exposed_set  = set(self.exposed)
        partial      = {}   # labelled positions -> the exposed categories seen on those positions only

        def is_exposed(key) -> bool:
            if not isinstance(key, tuple) or UNLABELED not in key:
                return key in exposed_set
            labelled = tuple(p for p, value in enumerate(key) if value != UNLABELED)
            if not labelled:
                return False
            if labelled not in partial:
                partial[labelled] = set(tuple(e[p] for p in labelled) for e in exposed_set)
            return tuple(key[p] for p in labelled) in partial[labelled]

        self.indices = [i for i, key in enumerate(self.keys) if is_exposed(key)]
        self.num_samples = int(len(self.indices) // self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = self.num_samples

    def __iter__(self) -> Iterable[int]:
        return iter(self._share(self.indices))

    def __len__(self) -> int:
        return self.num_selected_samples
