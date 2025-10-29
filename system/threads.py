from threading import Lock, RLock, Event, Condition
import threading
import traceback
import time
import os

class ThreadException(Exception): pass
# class _Thread:
#     def __init__(self, group=None, target=None, name=None, daemon=False, args=(), kwargs={}) -> None:
#         self.__added_time = time.time()
#         self.__exec_information = {
#             "group" : group,
#             "target": target,
#             "name"  : name,
#             "daemon": daemon,
#             "args"  : args,
#             "kwargs": kwargs
#         }
#         self.__local_lock = threading.Lock()
#         # Generate the thread id with 16 bytes.
#         # (2 ** 16 available thread id for each depth, former bytes comes from the parent thread id)
#         __id = os.urandom(2).hex()
#         self.return_val = None
# 
#     def __getitem__(self, key) -> "_Thread":
#         if key[:self.__depth*4] != self.__thread_id:
#             __parent = self.find_by_id(thread_id=key[:self.__depth*4]) # Find the parent thread.
#             return __parent[key[self.__depth*4:]]
#         else:
#             key = key[-4:] # Get the last 4 characters (16 bits) of the key.
#             return self.__children[key]
#         
#     def __setitem__(self, key, value) -> None:
#         if key[:self.__depth*4] != self.__thread_id:
#             __parent = self.find_by_id(thread_id=key[:self.__depth*4])
#             __parent[key[self.__depth*4:]] = value
#         else:
#             key = key[-4:]
#             self.__children[key] = value
# 
#     def find_by_id(self, thread_id:str) -> "_Thread":
#         __depth = len(thread_id) // 4 # The depth of the thread id Hexadecimal string.
#         if __depth > self.__head.__max_depth: raise KeyError("Thread id not found")
#         __thread = self.__head
#         for _i in range(0, len(thread_id), 4):
#             __thread = __thread[thread_id[_i:_i+4]]
#         return __thread
# 
#     def find_by_posix_id(self, posix_id:int) -> "_Thread":
#         __thread_id = self.__head.posix_id_to_thread_id.get(posix_id, None)
#         if __thread_id is None: raise KeyError("Thread id not found")
#         return self.find_by_id(thread_id=__thread_id)
#     
#     def get_thread_id_by_posix_id(self, posix_id:int) -> str:
#         return self.__head.posix_id_to_thread_id.get(posix_id, None)
# 
#     def run(self) -> None:
#         __th = threading.Thread(group =self.__exec_information["group"],
#                             target=self.__target_wrapper,
#                             name  =self.__exec_information["name"],
#                             daemon=self.__exec_information["daemon"]
#                          )
#         self.__head.posix_id_to_thread_id[__th.ident] = self.__thread_id
#         __th.start()
# 
#     def __target_wrapper(self) -> None:
#         self.running.set()
#         try:
#             self.return_val = self.__exec_information["target"](
#                                      *self.__exec_information["args"],
#                                     **self.__exec_information["kwargs"])
#         except Exception as e:
#             self.return_val = ThreadException(
#                                 f"Thread raised an exception: \n\t{str(e)}\n\t" + \
#                                 "\n\t".join(traceback.format_exc().split(sep="\n")))
#         self.running.clear()
#         self.done.set()
# 
#     def __eq__(self, other) -> bool:
#         if   isinstance(other, int): return self.__thread_id == other
#         elif isinstance(other, _Thread): return self.__thread_id == other.__thread_id
#         return False
#     
#     def __gt__(self, other) -> bool:
#         # For thread sorting, the thread with deeper and older thread is the first.
#         if self.__depth < other.__depth: return True
#         elif self.__depth == other.__depth: return self.__added_time > other.__added_time
#     
#     def __lt__(self, other) -> bool:
#         # For thread sorting, tne thread with deeper and older thread is the first.
#         if self.__depth > other.__depth: return True
#         elif self.__depth == other.__depth: return self.__added_time < other.__added_time
# 
#     def __str__(self) -> str:
#         return f"Thread( Thread id: {self.__thread_id} )"
    
# Singleton style Thread queue, which is used to manage the threads
class _Thread_Pool:
    '''
    Thread pool manager.
    This class is a singleton class.
    Manage the threads and thread queue, and return the result of the thread.
    Maximum number of threads can be set by set_max_thread method, automatically set to the number of CPU cores.
    '''
    __instance              = None
    __max_running_threads   = os.cpu_count()

    @classmethod
    def set_max_thread(cls, max_thread:int) -> None: cls.__max_running_threads= max_thread

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__is_initialized = False
        return cls.__instance
    def __init__(self) -> None:
        if self.__is_initialized: return
        self.__is_initialized       = True
        self.__posix_id_to_tid      = {}          # Map the POSIX thread id to the thread id.
        self.__thread_queue         = []          # Thread queue.
        self.__queue_cv             = Condition() # Condition variable for the thread queue.
        self.__state_cv             = Condition() # Condition variable for running limit.
        self.__run_flag             = Event()     # Event flag to stop the thread manager.
        self.__thread_manager       = None        # Thread manager thread.
        self.__run_counter          = 0           # Number of running threads.

    def set_max_thread(self, max_thread:int) -> None: self.__max_running_threads = max_thread
    def register(self, owner_id, group=None, target=None, name=None, daemon=False, args=(), kwargs={}) -> str:
        # Get the thread id of the owner thread. (Allow None)
        __owner_id = self.__posix_id_to_tid[owner_id] if owner_id in self.__posix_id_to_tid.keys() else ""
        surfix = os.urandom(2).hex() # Generate a random 4 character Hexadecimal string as the suffix of the thread id.
        __child_id = __owner_id + surfix if __owner_id else surfix
        self.__posix_id_to_tid[threading.get_ident()] = __child_id
        # If the thread manager is not started or finished, start the thread manager.
        if not self.__thread_manager or not self.__thread_manager.is_alive():
            self.__thread_manager = threading.Thread(target=self.manager)
            self.__thread_manager.start()
        return __child_id
    
    def enqueue(self, _th) -> None:
        with self.__queue_cv:
            self.__thread_queue.append(_th)
            self.__queue_cv.notify()

    # def notify(self) -> None:
    #     with self.__state_cv:
    #         self.__run_counter -= 1
    #         self.__state_cv.notify()

    def manager(self) -> None:
        while not self.__run_flag.is_set():
            try: 
                with self.__queue_cv:
                    if len(self.__thread_queue) == 0: 
                        self.__queue_cv.wait() # Wait for the thread queue to be not empty.
                        self.__thread_queue.sort() # Sort the thread queue by the thread depth and lifetime.
                    __th = self.__thread_queue.pop(0)
                self.__run_counter += 1
                __th.run() # Run the thread.
                # with self.__state_cv:
                #    if self.__run_counter >= self.__max_running_threads: self.__state_cv.wait() 
            except Exception as e: print(f"Thread manager raised an exception: {e}" + traceback.format_exc(), flush=True)

    def __del__(self):
        self.__run_flag.set()
        with self.__queue_cv:
            self.__thread_queue.clear()
            self.__queue_cv.notify()
        if self.__thread_manager and self.__thread_manager.is_alive(): self.__thread_manager.join()

class Thread:
    '''
    Thread class with return value.
    How many threads you start, the number of threads running at the same time is limited to the number of CPU cores.
    Abstract the thread management process.
    '''
    @classmethod
    def set_max_thread(cls, max_thread:int) -> None: _Thread_Pool.set_max_thread(max_thread)
    def __init__(self, group=None, target=None, name=None, daemon=False, args=(), kwargs={}) -> None:
        # Get the thread id of the current thread.
        self.__added_time   = time.time()
        self.__thread_pool  = _Thread_Pool()
        self.__owner        = threading.get_ident()
        self.id    : int    = self.__thread_pool.register(owner_id=self.__owner)
        self.__exec_information = {
            "group" : group,
            "target": target,
            "name"  : name,
            "daemon": daemon,
            "args"  : args,
            "kwargs": kwargs
        }
        self.running = threading.Event()
        self.done    = threading.Event()
        
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
                                f"Thread raised an exception: \n\t{str(e)}\n\t" + \
                                "\n\t".join(traceback.format_exc().split(sep="\n")))
        # self.__thread_pool.notify()
        self.done.set()
        self.running.clear()

    def run(self) -> None:
        __th = threading.Thread(
                            group =self.__exec_information["group"],
                            target=self.__exec_wrapper,
                            name  =self.__exec_information["name"],
                            daemon=self.__exec_information["daemon"])
        __th.start()
   
    def join(self):
        self.done.wait()
        if isinstance(self.return_val, ThreadException): raise self.return_val
        return self.return_val

    def __eq__(self, other) -> bool:
        if   isinstance(other, int):    return self.id == other.id
        elif isinstance(other, Thread): return self.id == other.id
        return False
    
    def __gt__(self, other) -> bool:
        # For thread sorting, the thread with deeper and older thread is the first.
        if len(self.id) < len(other.id):    return True
        elif len(self.id) == len(other.id): return self.__added_time > other.__added_time
    
    def __lt__(self, other) -> bool:
        # For thread sorting, tne thread with deeper and older thread is the first.
        if len(self.id) > len(other.id):    return True
        elif len(self.id) == len(other.id): return self.__added_time < other.__added_time
