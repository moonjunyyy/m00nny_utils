from threading import Lock, RLock, Event, Condition
import threading
import traceback
import time
import os

if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(
            path.dirname(
                path.dirname(
                    path.dirname(
                        path.abspath(__file__)
                    )))
        )
    from m00nny_utils.system.log import Log
else:
    from .log import Log

logger = Log("ThreadPool")


class TaskException(Exception):
    pass


class _Task_Pool:
    """
    Thread pool manager.
    This class is a singleton class.
    Manage the threads and thread queue, and return the result of the thread.
    Maximum number of threads can be set by set_max_thread method,
    automatically set to the number of CPU cores.
    """
    __instance = None
    __max_running_threads = os.cpu_count()

    @classmethod
    def set_max_thread(
        cls, max_thread: int
    ) -> None:
        cls.__max_running_threads = max_thread
        while len(cls.__instance.__workers) != cls.__max_running_threads:
            if len(cls.__instance.__workers) < cls.__max_running_threads:
                cls.__instance._expand_worker()
            else:
                cls.__instance._remove_worker()

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.ref_count = 0
            cls.__instance.__is_initialized = False
        cls.__instance.ref_count += 1
        return cls.__instance

    def __init__(self) -> None:
        if self.__is_initialized:
            return
        self.__is_initialized = True
        self.__task_queue = []         # Thread queue.
        self.__queue_cv = Condition()  # Condition variable for the task queue.
        self.__workers = {}            # Worker threads.
        self.__stop_event = Event()
        # Initialize the worker threads.
        self.set_max_thread(self.__max_running_threads)

    def _expand_worker(self) -> None:
        __worker = _Worker_Thread(self)
        __worker.start()
        self.__workers[__worker.ident] = __worker

    def _remove_worker(self) -> None:
        __id = self.__workers.keys()[0]
        __worker = self.__workers.get(__id, None)
        self.__workers.remove(__id)
        if __worker is not None:
            __worker.stop()
        # __worker.join() # No reason to wait, can cause deadlock.

    def register(
        self,
        group=None, target=None, name=None, daemon=False,
        args=(), kwargs={}
    ) -> str:
        # Check the thread identity
        _worker_id = threading.get_ident()
        _worker = self.__workers.get(_worker_id, None)
        if _worker is None:
            _owner_id = ""
        else:
            _owner_id = _worker.local.current_thread
        # Generate a random 4 character Hexadecimal string as the suffix.
        _surfix = os.urandom(2).hex()
        _child_id = f"{_owner_id}.{_surfix}" if _owner_id else _surfix
        return _child_id

    def enqueue(self, _th) -> None:
        with self.__queue_cv:
            self.__task_queue.append(_th)
            self.__queue_cv.notify()

    def dequeue(self):
        while True:
            try:
                with self.__queue_cv:
                    if len(self.__task_queue) == 0:
                        self.__queue_cv.wait(timeout=0.1)
                    # Sort the thread queue by the thread depth and lifetime.

                    self.__queue_cv.notify()
                    if self.__stop_event.is_set():
                        return None
                    else:
                        self.__task_queue.sort()
                        return self.__task_queue.pop(0)
            except:
                if self.__stop_event.is_set():
                    with self.__queue_cv:
                        self.__queue_cv.notify()
                    return None

    def release(self):
        self.ref_count -= 1
        if self.ref_count == 0:
            self.__stop_event.set()
            with self.__queue_cv:
                self.__queue_cv.notify_all()
            for __worker in self.__workers.values():
                __worker.join()


class _Worker_Thread(threading.Thread):
    def __init__(self, pool: _Task_Pool) -> None:
        super().__init__()
        self.local = threading.local()
        self.local.current_thread = None
        self.pool = pool

    def run(self) -> None:
        while True:
            try:
                __th = self.pool.dequeue()
                if __th is None:
                    break  # Stop the worker thread.
                self.local.current_thread = __th.id
                __th._exec_wrapper()
            except Exception as e:
                logger.error(
                    f"Worker thread raised an exception: {e}"
                    + traceback.format_exc()
                )
            finally:
                self.local.current_thread = None


class Task:
    """
    Thread class with return value.
    How many threads you start,
    the number of threads running at the same time is limited to the CPU cores.
    Abstract the thread management process.
    """
    @classmethod
    def set_max_thread(
        cls, max_thread: int) -> None: _Task_Pool.set_max_thread(max_thread)

    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        daemon=False,
        priority=5,
        args=(),
        kwargs={}
    ):
        # Get the thread id of the current thread.
        self.__added_time = time.time()
        self.__thread_pool = _Task_Pool()
        self.id: int = self.__thread_pool.register()
        self.priority = priority
        self.__exec_information = {
            "group": group,
            "target": target,
            "name": name,
            "daemon": daemon,
            "args": args,
            "kwargs": kwargs
        }
        self.running = threading.Event()
        self.done = threading.Event()

    def start(self) -> None:
        if not self.running.is_set() and not self.done.is_set():
            self.__thread_pool.enqueue(self)

    def _exec_wrapper(self) -> None:
        self.done.clear()
        self.running.set()
        try:
            self.return_val = self.__exec_information["target"](
                *self.__exec_information["args"],
                **self.__exec_information["kwargs"])
        except Exception as e:
            self.return_val = TaskException(
                f"Thread raised an exception: \n\t{str(e)}\n\t" +
                "\n\t".join(traceback.format_exc().split(sep="\n")))
        finally:
            self.running.clear()
            self.done.set()

    def join(self):
        self.done.wait()
        if isinstance(self.return_val, TaskException):
            raise self.return_val
        self.__thread_pool.release()
        return self.return_val

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            return self.id == other.id
        elif isinstance(other, Task):
            return self.id == other.id
        return False

    def __gt__(self, other) -> bool:
        # For thread sorting.
        # The thread with deeper and older thread is the first.
        if self.priority > other.priority:
            return True
        if len(self.id) < len(other.id):
            return True
        elif len(self.id) == len(other.id):
            return self.__added_time > other.__added_time

    def __lt__(self, other) -> bool:
        # For thread sorting.
        # The thread with deeper and older thread is the first.
        if self.priority < other.priority:
            return True
        if len(self.id) > len(other.id):
            return True
        elif len(self.id) == len(other.id):
            return self.__added_time < other.__added_time


if __name__ == '__main__':
    import random

    def test_func(x):
        time.sleep(random.uniform(0.1, 2.0))
        return x * x

    tasks = [
        Task(target=test_func, args=(i,))
        for i in range(os.cpu_count() * 6)
    ]
    for task in tasks:
        task.start()
    results = [(task.id, task.join())for task in tasks]
    print(results)
