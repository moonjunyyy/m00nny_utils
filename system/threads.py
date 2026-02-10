from threading import Lock, RLock, Event, Condition
import threading
import traceback
import time
import os


class ThreadException(Exception):
    pass

# Singleton style Thread queue, which is used to manage the threads


class _Thread_Pool:
    '''
    Thread class with return value.
    How many threads you start, the number of threads running at the same time is limited to the number of CPU cores.
    Abstract the thread management process.
    '''
    @classmethod
    def set_max_thread(
        cls, max_thread: int) -> None: _Thread_Pool.set_max_thread(max_thread)

    def __init__(self, group=None, target=None, name=None, daemon=False, args=(), kwargs={}) -> None:
        # Get the thread id of the current thread.
        self.__added_time = time.time()
        self.__thread_pool = _Thread_Pool()
        self.__owner = threading.get_ident()
        self.id: int = self.__thread_pool.register(owner_id=self.__owner)
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

    def __exec_wrapper(self) -> None:
        self.running.set()
        try:
            self.return_val = self.__exec_information["target"](
                *self.__exec_information["args"],
                **self.__exec_information["kwargs"])
        except Exception as e:
            self.return_val = ThreadException(
                f"Thread raised an exception: \n\t{str(e)}\n\t" +
                "\n\t".join(traceback.format_exc().split(sep="\n")))
        # self.__thread_pool.notify()
        self.done.set()
        self.running.clear()

    def run(self) -> None:
        __th = threading.Thread(
            group=self.__exec_information["group"],
            target=self.__exec_wrapper,
            name=self.__exec_information["name"],
            daemon=self.__exec_information["daemon"])
        __th.start()

    def join(self):
        self.done.wait()
        if isinstance(self.return_val, ThreadException):
            raise self.return_val
        return self.return_val

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            return self.id == other.id
        elif isinstance(other, Thread):
            return self.id == other.id
        return False

    def __gt__(self, other) -> bool:
        # For thread sorting, the thread with deeper and older thread is the first.
        if len(self.id) < len(other.id):
            return True
        elif len(self.id) == len(other.id):
            return self.__added_time > other.__added_time

    def __lt__(self, other) -> bool:
        # For thread sorting, tne thread with deeper and older thread is the first.
        if len(self.id) > len(other.id):
            return True
        elif len(self.id) == len(other.id):
            return self.__added_time < other.__added_time
