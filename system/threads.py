import threading
import traceback
import asyncio
import time
import os
    
class ThreadException(Exception): pass
class _Thread:
    __threads = {}
    def __init__(self, parent_thread:"_Thread"=None, group=None, target=None, name=None, daemon=False, args=(), kwargs={}) -> None:
        self.__added_time = time.time()
        self.__children   = {}
        self.__exec_information = {
            "group" : group,
            "target": target,
            "name"  : name,
            "daemon": daemon,
            "args"  : args,
            "kwargs": kwargs
        }
        if parent_thread is None:
            self.__depth         = 0
            self.__max_depth     = 0
            self.__thread_id     = ""
            self.__parent_thread = None
            self.__is_head       = True
            self.__head          = self
            self.posix_id_to_thread_id = {threading.get_ident(): ""} # Set the main thread id.
            return
        self.running    = False # RAW problem does not so much matter.
        self.done       = False # RAW problem does not so much matter.
        self.__depth = parent_thread.__depth + 1
        # Generate the thread id with 16 bytes.
        # (2 ** 16 available thread id for each depth, former bytes comes from the parent thread id)
        __id = os.urandom(2).hex()
        self.__parent_thread = parent_thread
        self.__thread_id     = parent_thread.__thread_id + __id
        self.__head          = parent_thread.__head
        self.__parent_thread[__id] = self
        if self.__depth > self.__head.__max_depth: self.__head.__max_depth = self.__depth
        self.__class__.__threads[self.__thread_id] = self
        self.__is_head = False
        self.return_val = None

    def __getitem__(self, key) -> "_Thread":
        if key[:self.__depth*4] != self.__thread_id:
            __parent = self.find_by_id(thread_id=key[:self.__depth*4]) # Find the parent thread.
            return __parent[key[self.__depth*4:]]
        else:
            key = key[-4:] # Get the last 4 characters (16 bits) of the key.
            return self.__children[key]
        
    def __setitem__(self, key, value) -> None:
        if key[:self.__depth*4] != self.__thread_id:
            __parent = self.find_by_id(thread_id=key[:self.__depth*4])
            __parent[key[self.__depth*4:]] = value
        else:
            key = key[-4:]
            self.__children[key] = value

    def find_by_id(self, thread_id:str) -> "_Thread":
        __depth = len(thread_id) // 4 # The depth of the thread id Hexadecimal string.
        if __depth > self.__head.__max_depth: raise KeyError("Thread id not found")
        __thread = self.__head
        for _i in range(0, len(thread_id), 4):
            __thread = __thread[thread_id[_i:_i+4]]
        return __thread

    def find_by_posix_id(self, posix_id:int) -> "_Thread":
        __thread_id = self.__head.posix_id_to_thread_id.get(posix_id, None)
        if __thread_id is None: raise KeyError("Thread id not found")
        return self.find_by_id(thread_id=__thread_id)
    
    def get_thread_id_by_posix_id(self, posix_id:int) -> str:
        return self.__head.posix_id_to_thread_id.get(posix_id, None)

    def run(self) -> None:
        __th = threading.Thread(group =self.__exec_information["group"],
                            target=self.__target_wrapper,
                            name  =self.__exec_information["name"],
                            daemon=self.__exec_information["daemon"]
                            )
        self.__head.posix_id_to_thread_id[__th.ident] = self.__thread_id
        __th.start()
    
    def join(self):
        while not self.done: asyncio.run(asyncio.sleep(0.01))
        return self.return_val

    def __target_wrapper(self) -> None:
        self.running    = True
        try:
            self.return_val = self.__exec_information["target"](
                                     *self.__exec_information["args"],
                                    **self.__exec_information["kwargs"])
        except Exception as e:
            self.return_val = ThreadException(
                                f"Thread raised an exception: \n\t{str(e)}\n\t" + \
                                "\n\t".join(traceback.format_exc().split(sep="\n")))
        self.running    = False
        self.done       = True

    def __eq__(self, other) -> bool:
        if   isinstance(other, int):               return self.__thread_id == other
        elif isinstance(other, _Thread): return self.__thread_id == other.__thread_id
        return False
    
    def __gt__(self, other) -> bool:
        # For thread sorting, the thread with the lower, newer depth is the last.
        if self.__depth != other.__depth: return self.__depth      < other.__depth
        else:                         return self.__added_time > other.__added_time
    
    def __lt__(self, other) -> bool:
        # For thread sorting, the thread with the higher, older depth is the first.
        if self.__depth != other.__depth: return self.__depth      > other.__depth
        else:                         return self.__added_time < other.__added_time
    
    def add_child(self, thread_id:str=None, group=None, target=None, name=None, daemon=False, args=(), kwargs={}) -> str:
        if thread_id is None:
            _th = _Thread(parent_thread=self,    group=group, target=target, name=name, daemon=daemon, args=args, kwargs=kwargs)
        else:
            _parent = self.find_by_id(thread_id=thread_id)
            _th = _Thread(parent_thread=_parent, group=group, target=target, name=name, daemon=daemon, args=args, kwargs=kwargs)
        return _th.__thread_id
    
    def remove_child(self, thread_id:str) -> None:
        if thread_id[:self.__depth*4] != self.__thread_id:
            _parent = self.find_by_id(thread_id=thread_id[:self.__depth*4])
            _parent.remove_child(thread_id=thread_id)
        else: self.__children.pop(thread_id[self.__depth*4:], None)

    def is_empty(self) -> bool:
        return len(self.__children) == 0

    @property
    def num_waiting(self) -> int:
        if self.__is_head: return sum([not (_th.running or _th.done) for _th in self.__threads.values()])
    
    @property
    def num_running(self) -> int:
        if self.__is_head: return sum([_th.running for _th in self.__threads.values()])

    @property
    def num_done(self) -> int:
        if self.__is_head: return sum([_th.done for _th in self.__threads.values()])

    def __iter__(self):
        if self.__is_head: return iter(list(self.__class__.__threads.values()).sort())
        return self.__head.__iter__()
    
    def __str__(self) -> str:
        return f"Thread id: {self.__thread_id}, Depth: {self.__depth}, Max depth: {self.__max_depth}"
    
# class _Thread(threading.Thread):
#     '''
#     Thread implementation with return value.
#     This class is actually a wrapper of the threading.Thread class.
#     '''
#     def __init__(self, group=None, target=None, name=None, daemon=False, args=(), kwargs={}) -> None:
#         super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
#         self._return = None
    
#     def run(self) -> None:
#         try:
#             if self._target is not None: self._return = self._target(*self._args, **self._kwargs)
#         except Exception as e: self._return = ThreadException(f"Thread raised an exception: {str(e)}\n" + traceback.format_exc())

#     def join(self, *args):
#         super().join(*args)
#         return self._return

# Singleton style Thread queue, which is used to manage the threads
class _Thread_Pool:
    '''
    Thread pool manager.
    This class is a singleton class.
    Manage the threads and thread queue, and return the result of the thread.
    Maximum number of threads can be set by set_max_thread method, automatically set to the number of CPU cores.
    '''
    __instance = None
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance
    
    def __init__(self) -> None:
        if not self.__initialized:
            self.__lock               = threading.Lock()
            self.__thread_queue       = []
            self.__master_thread      = _Thread()
            self.__running_thread_max = os.cpu_count() * 2 # Set the maximum number of threads to the number of CPU cores.
            self.__thread_manager     = None
            self.__initialized        = True

    def set_max_thread(self, max_thread:int) -> None:
        self.__running_thread_max = max_thread

    def register(self, owner_id, group=None, target=None, name=None, daemon=False, args=(), kwargs={}) -> str:
        with self.__lock:
            __owner_id = self.__master_thread.get_thread_id_by_posix_id(owner_id) # Get the thread id of the owner thread. (Allow None)
            __child_id = self.__master_thread.add_child(thread_id=__owner_id, group=group, target=target, name=name, daemon=daemon, args=args, kwargs=kwargs)
            if not self.__thread_manager or not self.__thread_manager.is_alive():
                # If the thread manager is not started or finished, start the thread manager.
                self.__thread_manager = threading.Thread(target=self.manager, daemon=True)
                self.__thread_manager.start()
        return __child_id
    
    def enqueue(self, id) -> None:
        with self.__lock:
            self.__thread_queue.append(self.__master_thread.find_by_id(thread_id=id))

    def wait(self, id) -> None:
        self.__master_thread.find_by_id(thread_id=id).join()

    def manager(self) -> None:
        while not self.__master_thread.is_empty():
            try:
                asyncio.run(asyncio.sleep(0.01))
                # If too many job is waiting, allow the running threads to be more than the maximum.
                if self.__master_thread.num_running > \
                    self.__running_thread_max + (self.__master_thread.num_waiting / (self.__running_thread_max * 2)): continue
                if len(self.__thread_queue) == 0: continue
                with self.__lock:
                    self.__thread_queue.sort() # Sort the thread queue by the thread depth and lifetime.
                    __th = self.__thread_queue.pop(0)
                    __th.run()
            except Exception as e: print(f"Thread manager raised an exception: {e}" + traceback.format_exc())

class Thread:
    '''
    Thread class with return value.
    How many threads you start, the number of threads running at the same time is limited to the number of CPU cores.
    Abstract the thread management process.
    '''
    def __init__(self, group=None, target=None, name=None, daemon=False, args=(), kwargs={}) -> None:
        self.thread_pool  = _Thread_Pool()
        # Get the thread id of the current thread.
        self.__owner      = threading.get_ident()
        self.id    : int  = self.thread_pool.register(owner_id=self.__owner, group=group, target=target, name=name, daemon=daemon, args=args, kwargs=kwargs)
        self.state : bool = False

    def start(self) -> None:
        if self.state: raise ThreadException("Thread already started")
        self.thread_pool.enqueue(id=self.id)
        self.state = True

    def join(self):
        if not self.state: raise ThreadException("Thread not started")
        __return = self.thread_pool.wait(id=self.id)
        self.state = False
        return __return