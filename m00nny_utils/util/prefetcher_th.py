import torch
import threading
from torch import Tensor
from typing import List, Tuple, Dict, Any, TypeVar, Sequence, Callable


_D = TypeVar('_D')   # Data component (List, Tuple, Dict, Any)


def _build_batch(
        batch: List[Any,]
) -> Tuple[Tensor,] | List[Tensor,] | Dict[Any, Tensor]:
    typename = None
    typename = type(batch[0])
    if typename is None:
        raise RuntimeError(
            f"build_batch : Error occured building batch : type is {typename}")
    elif typename is list or typename is tuple:
        try:
            batch = list(zip(*batch))
            batch = [torch.stack(b) if isinstance(
                b[0], Tensor) else torch.tensor(b) for b in batch]
        except Exception as e:
            print(f"build_batch : Error occured building batch\n{e}")
    elif typename is dict:
        try:
            keys = list(batch[0].keys())
            batch = {k: torch.stack([s[k] for s in batch]) if isinstance(
                batch[0][k], Tensor) else torch.tensor(
                    [s[k] for s in batch]
            ) for k in keys}
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
    def __init__(self, num_workers: int, name: str = 'Prefetcher'):
        self.num_workers = num_workers
        self.closed = threading.Event()
        self.n_iter = 1
        self.name = name
        self.ports = []
        self.workers = []
        self.output_buffer = []
        self.output_lock = threading.Lock()
        self.output_cv = threading.Condition(self.output_lock)
        self.flush_event = threading.Event()
        self.batch_size = 0
        self._worker_cursor = 0
        self._none_count = 0
        self.dots = 3
        self.fetch_thread = None

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
            for b in range(self.parent.batch_size):
                sample = self.parent.get_worker().get_sample()
                if sample is None:
                    self.parent._none_count += 1
                    if self.parent._none_count == len(self.parent.workers):
                        break
                else:
                    batch.append(sample)
            if len(batch) == 0:
                raise StopIteration
            batch = _build_batch(batch)
            return batch

    def load(
        self,
        dataset: Sequence[Any],
        transform: Callable[[Any,], Tuple[Tensor, ...]],
        batch_size: int,
        n_iter: int = 1,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler = None,
        generator: torch.Generator = None
    ) -> None:
        if sampler is None:
            if generator is None:
                self.generator = torch.Generator()
            else:
                generator = generator
            if shuffle:
                sampler = torch.utils.data.RandomSampler(
                    dataset, generator=self.generator)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)
        else:
            assert not shuffle, 'Cannot use a custom sampler and random = True'
        self.batch_size = batch_size
        self.dataset = dataset
        self.transform = transform
        self.sampler = sampler
        self.n_iter = n_iter
        self._worker_cursor = 0
        self.indices = list(self.sampler)

    def get_worker(self):
        worker = self.workers[self._worker_cursor]
        self._worker_cursor += 1
        self._worker_cursor %= self.num_workers
        return worker

    def create_workers(self):
        self._none_count = 0
        for n in range(self.num_workers):
            worker = _PrefetchWorker(
                self.dataset,
                self.transform,
                self.indices[n::self.num_workers],
                buffer_len=(self.batch_size // self.num_workers + 1) * 2
            )
            worker.run()
            self.workers.append(worker)

    def stop_workers(self):
        for worker in self.workers:
            worker.close()
        self.workers = []

    def __len__(self):
        return (
            len(self.sampler) // self.batch_size +
            int(len(self.sampler) % self.batch_size != 0)
        )

    def __iter__(self):
        self.stop_workers()
        self.create_workers()
        return Prefetcher.__PrefetcherIterater(self)

    def __del__(self):
        for worker in self.workers:
            worker.close()
