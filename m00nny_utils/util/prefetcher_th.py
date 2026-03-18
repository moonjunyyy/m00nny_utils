import torch
import threading
import numpy as np
from torch import Tensor
from ..sampler import Sampler
from typing import List, Tuple, Dict, Any, TypeVar, Sequence, Callable


_D = TypeVar('_D')   # Data component (List, Tuple, Dict, Any)


def _list_composer(
    batch: List[Any,]
) -> List[Tensor,]:
    sample_type = type(batch[0])
    if sample_type is Tensor:
        batch = torch.stack(batch)
    elif sample_type in [int, float, bool, np.array]:
        batch = torch.tensor(batch)
    return batch


def _compose_batch(
    batch: List[Any,]
) -> Tuple[Tensor,] | List[Tensor,] | Dict[Any, Tensor]:
    typename = None
    typename = type(batch[0])
    if typename is None:
        raise RuntimeError(
            f"build_batch : Error occured building batch : type is {typename}"
        )
    elif typename is list or typename is tuple:
        try:
            batch = list(zip(*batch))
            batch = [_list_composer(b) for b in batch]
        except Exception as e:
            print(f"build_batch : Error occured building batch\n{e}")
    elif typename is dict:
        try:
            keys = list(batch[0].keys())
            batch = {
                k: _list_composer([sample[k] for sample in batch])
                for k in keys
            }
        except Exception as e:
            print(f"build_batch : Error occured building batch\n{e}")
    else:
        batch = torch.stack(batch)
    return batch


class _PrefetchWorker:
    def __init__(
        self,
        dataset,
        transform,
        indices,
        buffer_len=16,
        daemon: bool = True
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.daemon = daemon
        self.output_buffer_lock = threading.Lock()
        self.done_flag = threading.Event()
        self.output_empty_cv = threading.Condition(self.output_buffer_lock)
        self.output_full_cv = threading.Condition(self.output_buffer_lock)
        self.output_buffer = []
        self.buffer_len = buffer_len
        self.done_flag.clear()

    def get_sample(self):
        with self.output_empty_cv:
            if len(self.output_buffer) == 0:
                if self.done_flag.is_set():
                    sample = None
                else:
                    self.output_empty_cv.wait()
                    sample = self.output_buffer.pop(0)
            else:
                sample = self.output_buffer.pop(0)
            self.output_full_cv.notify()
        return sample

    def run(self):
        self.th = threading.Thread(
            target=self._loop,
            daemon=self.daemon
        )
        self.th.start()

    def _loop(self):
        for index in self.indices:
            if self.done_flag.is_set():
                return
            sample = self.dataset[index]
            if self.transform is not None:
                sample = self.transform(sample)
            with self.output_full_cv:
                if len(self.output_buffer) >= self.buffer_len:
                    self.output_full_cv.wait()
                self.output_buffer.append(sample)
                self.output_empty_cv.notify()
        self.done_flag.set()
        with self.output_empty_cv:
            self.output_empty_cv.notify()

    def close(self):
        self.done_flag.set()
        with self.output_empty_cv:
            self.output_empty_cv.notify()
        with self.output_full_cv:
            self.output_full_cv.notify()

    def __del__(self):
        self.close()


class Prefetcher:
    def __init__(
        self,
        num_workers: int,
        name: str = 'Prefetcher',
        *args,
        dataset: Sequence[Any] = None,
        transform: Callable[[Any,], Tuple[Tensor, ...]] = None,
        batchsize: int = 0,
        n_iter: int = 1,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler = None,
        generator: torch.Generator = None,
        device: torch.device = None,
        pin_memory: bool = False,
        **kargs

    ):
        self.num_workers = num_workers
        self.n_iter = 1
        self.name = name
        self.workers = []
        self.output_buffer = []
        self.batchsize = 0
        self._worker_cursor = 0
        self._none_count = 0

        self.dataset = dataset
        self.transform = transform
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.sampler = sampler
        self.generator = generator

        self.device = device
        self.pin_memory = pin_memory

        self.load(
            dataset=dataset,
            transform=transform,
            batchsize=batchsize,
            n_iter=n_iter,
            shuffle=shuffle,
            sampler=sampler,
            generator=generator
        )

    class __PrefetcherIterater:
        def __init__(self, parent):
            self.parent = parent

        def __next__(self):
            batch = []
            if self.parent._none_count >= len(self.parent.workers):
                for w in self.parent.workers:
                    w.close()
                self.parent.workers = []
                raise StopIteration
            for b in range(self.parent.batchsize):
                sample = self.parent._get_worker().get_sample()
                if sample is None:
                    self.parent._none_count += 1
                    if self.parent._none_count == len(self.parent.workers):
                        break
                else:
                    batch.append(sample)
            if len(batch) == 0:
                raise StopIteration
            batch = _compose_batch(batch)
            return batch

    def load(
        self,
        dataset: Sequence[Any],
        transform: Callable[[Any,], Tuple[Tensor, ...]],
        batchsize: int,
        n_iter: int = 1,
        shuffle: bool = False,
        sampler: Sampler = None,
        generator: torch.Generator = None,
        device: torch.device = None,
        pin_memory: bool = False
    ) -> None:
        self.batchsize = batchsize
        self.dataset = dataset
        self.transform = transform

        if sampler is None:
            if generator is None:
                self.generator = torch.Generator()
            else:
                generator = generator
            sampler = Sampler(
                dataset=dataset,
                shuffle=shuffle,
                generator=generator
            )
        else:
            assert not shuffle, 'Cannot use a custom sampler and random = True'
        self.sampler = sampler
        self.n_iter = n_iter
        self._worker_cursor = 0
        self.indices = list(self.sampler)

    def _get_worker(self):
        _worker = self.workers[self._worker_cursor]
        self._worker_cursor += 1
        self._worker_cursor %= self.num_workers
        return _worker

    def _create_workers(self):
        self._none_count = 0
        for n in range(self.num_workers):
            _worker = _PrefetchWorker(
                self.dataset,
                self.transform,
                self.indices[n::self.num_workers],
                buffer_len=(self.batchsize // self.num_workers + 1) * 2
            )
            _worker.run()
            self.workers.append(_worker)

    def _stop_workers(self):
        for _worker in self.workers:
            _worker.close()
        self.workers = []

    def __len__(self):
        return (
            len(self.sampler) // self.batchsize +
            int(len(self.sampler) % self.batchsize != 0)
        )

    def __iter__(self):
        self._stop_workers()
        self._create_workers()
        return Prefetcher.__PrefetcherIterater(self)

    def __del__(self):
        for worker in self.workers:
            worker.close()
