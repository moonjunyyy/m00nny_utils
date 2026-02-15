import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from typing import Optional, Sized, Iterable, Tuple, Union
from ..datasets.DomainDataset import DomainDataset


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


class OnlineSampler(Sampler):
    def __init__(
            self,
            data_source: Sized,
            num_tasks: int,
            m: int,
            n: int,
            rnd_seed: int,
            cur_iter: int= 0,
            varing_NM: bool= False,
            num_replicas: int=None,
            rank: int=None
    ) -> None:

        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.generator  = torch.Generator().manual_seed(rnd_seed)

        self.n  = n
        self.m  = m
        self.varing_NM = varing_NM
        self.task = cur_iter

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        self.disjoint_num   = len(self.classes) * n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        if not self.varing_NM:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred]
                blurred = blurred[num_blurred:]

            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
        else:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes = class_order[:self.disjoint_num].tolist()
            if self.disjoint_num > 0:
                self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            if self.blurry_num > 0:
                self.blurry_slice = [0] + torch.randint(0, self.blurry_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.blurry_num]
                self.blurry_classes = [class_order[self.disjoint_num + self.blurry_slice[i]:self.disjoint_num + self.blurry_slice[i + 1]].tolist() for i in range(num_tasks)]
            else:
                self.blurry_classes = [[] for _ in range(num_tasks)]
            # self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            # self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)

            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        num_blurred += 1
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            num_blurred = num_blurred * m // 100
            if num_blurred > 0:
                num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

                for i in range(num_tasks):
                    blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                    self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
                blurred = torch.tensor(blurred)
                blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
                print("blurry indices: ", len(blurred))
                # num_blurred = len(blurred) // num_tasks
                for i in range(num_tasks):
                    self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                    blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]

            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices[self.task])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int)-> None:
        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def get_task(self, cur_iter: int)-> Iterable[int]:
        indices = self.indices[cur_iter][self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]

class OnlineBatchSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], num_tasks: int, m: int, n: int, rnd_seed: int, batchsize: int=16, online_iter: int=1, cur_iter: int=0, varing_NM: bool=False, num_replicas: int=None, rank: int=None) -> None:
        super().__init__(data_source)
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.num_tasks  = num_tasks
        self.m      = m
        self.n      = n
        self.rnd_seed   = rnd_seed
        self.batchsize  = batchsize
        self.online_iter    = online_iter
        self.cur_iter   = cur_iter
        self.varing_NM  = varing_NM

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        self.disjoint_num   = len(self.classes) * n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        if not self.varing_NM:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred]
                blurred = blurred[num_blurred:]

            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)]
                num_batches     = int(self.indices[i].size(0) // self.batchsize)
                rest            = self.indices[i].size(0) % self.batchsize
                self.indices[i] = self.indices[i][:num_batches * self.batchsize].reshape(-1, self.batchsize).repeat(self.online_iter, 1).flatten().tolist() + self.indices[i][-rest:].tolist()
        else:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num].tolist()
            if self.disjoint_num > 0:
                self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)

            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        num_blurred += 1
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            num_blurred = num_blurred * m // 100
            num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            # num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]

            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
                num_batches     = int(self.indices[i].size(0) // self.batchsize)
                rest            = self.indices[i].size(0) % self.batchsize
                self.indices[i] = self.indices[i][:num_batches * self.batchsize].reshape(-1, self.batchsize).repeat(self.online_iter, 1).flatten().tolist() + self.indices[i][-rest:].tolist()

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices[self.task])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int) -> None:

        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def get_task(self, cur_iter : int) -> Iterable[int]:
        indices = self.indices[cur_iter][self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]

    def get_task_classes(self, cur_iter : int) -> Iterable[int]:
        return list(set(self.classes[self.indices[cur_iter]]))

class OnlineTestSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], exposed_class : Iterable[int], num_replicas: int=None, rank: int=None) -> None:
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.exposed_class  = exposed_class
        self.indices    = [i for i in range(self.data_source.__len__()) if self.targets[i] in self.exposed_class]

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        if self.distributed:
            self.num_samples = int(len(self.indices) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices))

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices)

    def __len__(self) -> int:
        return self.num_selected_samples


class OnlineSampler(Sampler):
    def __init__(self, data_source: Sized, num_tasks: int, m: int, n: int,  rnd_seed: int, cur_iter: int= 0, varing_NM: bool= False, num_replicas: int=None, rank: int=None) -> None:

        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.generator  = torch.Generator().manual_seed(rnd_seed)

        self.n  = n
        self.m  = m
        self.varing_NM = varing_NM
        self.task = cur_iter

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        self.disjoint_num   = len(self.classes) * n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        if not self.varing_NM:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred]
                blurred = blurred[num_blurred:]

            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
        else:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes = class_order[:self.disjoint_num].tolist()
            if self.disjoint_num > 0:
                self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            if self.blurry_num > 0:
                self.blurry_slice = [0] + torch.randint(0, self.blurry_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.blurry_num]
                self.blurry_classes = [class_order[self.disjoint_num + self.blurry_slice[i]:self.disjoint_num + self.blurry_slice[i + 1]].tolist() for i in range(num_tasks)]
            else:
                self.blurry_classes = [[] for _ in range(num_tasks)]
            # self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            # self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)

            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        num_blurred += 1
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            num_blurred = num_blurred * m // 100
            if num_blurred > 0:
                num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

                for i in range(num_tasks):
                    blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                    self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
                blurred = torch.tensor(blurred)
                blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
                print("blurry indices: ", len(blurred))
                # num_blurred = len(blurred) // num_tasks
                for i in range(num_tasks):
                    self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                    blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]

            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices[self.task])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int)-> None:
        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def get_task(self, cur_iter: int)-> Iterable[int]:
        indices = self.indices[cur_iter][self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]

class OnlineDomainSampler(Sampler):
    def __init__(
            self,
            data_source: DomainDataset,
            mode:str,
            num_tasks: int,
            m: int,
            n: int,
            rnd_seed: int,
            varing_NM: bool=False,
            online_iter: int=1,
            batchsize:int=64,
            cur_iter: int=0,
            num_replicas: int=None,
            rank: int=None
    ) -> None:
        super().__init__(data_source)
        if mode not in ['joint', 'cil', 'dil', 'vil']:
            raise ValueError(
                "mode should be one of ['joint', 'cil', 'dil', 'vil']"
            )
        self.data_source  = data_source
        self.domains      = self.data_source.domains
        self.targets      = self.data_source.targets
        self.classes      = self.data_source.classes
        self.domain_names = self.data_source.domain_names
        self.mode         = mode
        self.num_tasks    = num_tasks
        self.m            = m
        self.n            = n
        self.rnd_seed     = rnd_seed
        self.online_iter  = online_iter
        self.task         = cur_iter
        self.varing_NM    = varing_NM
        self.batchsize    = batchsize
        self.generator    = torch.Generator().manual_seed(rnd_seed) if rnd_seed is not None else torch.Generator()

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()
        self.distributed  = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank         = rank if rank is not None else 0

        if self.mode == 'joint':
            self.num_categories = len(self.classes)
            self.n = 0
            self.m = 0
        elif self.mode == 'cil': self.num_categories = len(self.classes)
        elif self.mode == 'dil': self.num_categories = len(self.domain_names)
        elif self.mode == 'vil': self.num_categories = len(self.classes) * len(self.domain_names)

        self.disjoint_num   = self.num_categories * self.n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = self.num_categories - self.disjoint_num
        self.blurry_num     = int(self.blurry_num   // num_tasks) * num_tasks

        if not self.varing_NM:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(self.num_categories, generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            if self.mode == 'joint':
                print(f"classes : {self.classes}")
            elif self.mode == 'cil':
                _pretty_print_nested_lists_in_double_columns(self.disjoint_classes, self.blurry_classes, "disjoint classes", "blurry classes", sep=" | ")
            elif self.mode == 'dil':
                _pretty_print_nested_lists_in_double_columns(self.disjoint_classes, self.blurry_classes, "disjoint classes", "blurry classes", sep=" | ")
            elif self.mode == 'vil':
                self.targets = [ta + (do * len(self.classes)) for ta, do in zip(self.targets, self.domains)]
                _pretty_print_nested_lists_in_double_columns(
                    [list(f"D{c//len(self.classes)}C{c%len(self.classes)}" for c in disjoint_task) for disjoint_task in self.disjoint_classes],
                    [list(f"D{c//len(self.classes)}C{c%len(self.classes)}" for c in blurry_task)   for blurry_task   in self.blurry_classes],
                    "disjoint classes", "blurry classes", sep=" | ")

            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if   self.targets[i] in self.disjoint_classes[j]: self.disjoint_indices[j].append(i)
                    elif self.targets[i] in self.blurry_classes[j]:   self.blurry_indices[j].append(i)

            num_blurreds = [len(self.blurry_indices[i]) * m // 100 for i in range(num_tasks)]
            # Randomly shuffle M% of blurry indices
            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurreds[i]]
                blurred = blurred[num_blurreds[i]:]

            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d, online_iter : %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i]), self.online_iter))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)]#.tolist()
                num_batches     = int(self.indices[i].size(0) // self.batchsize)
                rest            = self.indices[i].size(0) % self.batchsize
                self.indices[i] = self.indices[i][:num_batches * self.batchsize].reshape(-1, self.batchsize).repeat(1,self.online_iter).flatten().tolist() + self.indices[i][-rest:].repeat(int(self.online_iter)).tolist()
                print(f" -> {self.indices[i][:int(self.batchsize * self.online_iter)]}")
        else:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order = torch.randperm(self.num_categories, generator=self.generator)
            self.disjoint_classes = class_order[:self.disjoint_num].tolist()

            if self.disjoint_num > 0:
                self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            if self.mode == 'joint':
                print(f"classes : {self.classes}")
            elif self.mode == 'cil':
                _pretty_print_nested_lists_in_double_columns(self.disjoint_classes, self.blurry_classes, "disjoint classes", "blurry classes", sep=" | ")
            elif self.mode == 'dil':
                _pretty_print_nested_lists_in_double_columns(self.disjoint_classes, self.blurry_classes, "disjoint classes", "blurry classes", sep=" | ")
            elif self.mode == 'vil':
                self.targets = [ta + (do * len(self.classes)) for ta, do in zip(self.targets, self.domains)]
                _pretty_print_nested_lists_in_double_columns(
                    [list(f"D{c//len(self.classes)}C{c%len(self.classes)}" for c in disjoint_task) for disjoint_task in self.disjoint_classes],
                    [list(f"D{c//len(self.classes)}C{c%len(self.classes)}" for c in blurry_task)   for blurry_task   in self.blurry_classes],
                    "disjoint classes", "blurry classes", sep=" | ")

            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                    if self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices  [j].append(i)
                        num_blurred += 1

            # Randomly shuffle M% of blurry indices
            if self.m != 0:
                blurred = []
                num_blurred = num_blurred * self.m // 100
                num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

                for i in range(num_tasks):
                    blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                    self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
                blurred = torch.tensor(blurred)
                blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
                print("blurry indices: ", len(blurred))
                # num_blurred = len(blurred) // num_tasks
                for i in range(num_tasks):
                    self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                    blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]

            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)]#.tolist()
                num_batches     = int(self.indices[i].size(0) // self.batchsize)
                rest            = self.indices[i].size(0) % self.batchsize
                self.indices[i] = self.indices[i][:-rest].reshape(-1, int(self.batchsize)).repeat(1, int(self.online_iter)).flatten().tolist() + self.indices[i][-rest:].repeat(int(self.online_iter)).tolist()

    def _split_disjoint_blurry_categories(self,
                                          categories                      :Union[int, Iterable[int]],
                                          disjoint_class_ratio            :float,
                                          num_tasks                       :int,
                                          is_disjoint_class_ratio_constant:bool=True):
        if isinstance(categories, int): categories = list(range(categories))
        else: categories = list(categories)

        _category_order     = torch.randperm(len(categories), generator=self.generator)
        _n_disjoint_classes = int(len(categories) * disjoint_class_ratio)
        disjoint_classes    = category_order[:_n_disjoint_classes].tolist()
        blurry_classes      = category_order[_n_disjoint_classes:].tolist()

        for i in range(num_tasks):
            disjoint_num = int(len(categories) * disjoint_class_ratio[i])
            disjoint_classes.append(category_order[:disjoint_num].tolist())
            blurry_classes.append(category_order[disjoint_num:].reshape(-1).tolist())

        category_order   = torch.randperm(self.num_categories, generator=self.generator)
        return disjoint_classes, blurry_classes

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices[self.task])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int) -> None:
        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def get_task(self, cur_iter : int) -> Iterable[int]:
        indices = self.indices[cur_iter][self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]

    def get_task_classes(self, cur_iter : int) -> Iterable[int]:
        return list(set(self.classes[self.indices[cur_iter]]))

class OnlineTestSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], exposed_class : Iterable[int], num_replicas: int=None, rank: int=None) -> None:
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.exposed_class  = exposed_class
        self.indices    = [i for i in range(self.data_source.__len__()) if self.targets[i] in self.exposed_class]

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        if self.distributed:
            self.num_samples = int(len(self.indices) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices))

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices)

    def __len__(self) -> int:
        return self.num_selected_samples


class OnlineDomainTestSampler(Sampler):
    def __init__(
            self,
            data_source: Optional[Sized],
            exposed_domain_class: Iterable[Tuple[int, int]],
            mode,
            num_replicas: int = None,
            rank: int = None
    ) -> None:
        self.data_source = data_source
        self.targets = self.data_source.targets
        self.classes = self.data_source.classes
        self.domains = self.data_source.domains
        self.domain_names = self.data_source.domain_names
        self.mode = mode
        if mode not in ['joint', 'cil', 'dil', 'vil']:
            raise ValueError(
                "mode should be one of ['joint', 'cil', 'dil', 'vil']"
            )
        self.exposed_domain_class = exposed_domain_class
        self.exposed_classes = list(
            set([c for c, d in self.exposed_domain_class])
        )
        self.targets = self.data_source.targets
        self.domains = self.data_source.domains
        if self.mode == 'joint':
            pass
        else:
            self.targets = self.targets
        self.indices = [
            i for i in range(self.data_source.__len__())
            if (
                self.targets[i], self.domains[i]
            ) in self.exposed_domain_class or (
                self.domains[i] == -1 and
                self.targets[i] in self.exposed_classes
            )
        ]

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError(
                    "Distibuted package is not available, "
                    "but you are trying to use it."
                )
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError(
                    "Distibuted package is not available, "
                    "but you are trying to use it."
                )
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        if self.distributed:
            self.num_samples = int(len(self.indices) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(
                len(self.indices) // self.num_replicas
            )
        else:
            self.num_samples = int(len(self.indices))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices))

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices)

    def __len__(self) -> int:
        return self.num_selected_samples

