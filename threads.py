import threading
import traceback
import asyncio
import os

# Singleton style Thread queue, which is used to manage the threads
class _Thread_Pool:
    def set_max_thread(self, max_thread:int):
        _cls = type(self)
        _cls._instance._running_thread_max = max_thread

    def _register(self, group=None, target=None, name=None, daemon=False, args=(), kwargs={}):
        _cls = type(self)
        _id = os.urandom(32).hex()
        with _cls._instance._lock:
            _cls._instance._thread_dict[_id] = (group, target, name, daemon, args, kwargs)
            if not _cls._instance._thread_manager or not _cls._instance._thread_manager.is_alive():
                _cls._instance._thread_manager = threading.Thread(target=_cls._manager, args=(), daemon=True)
                _cls._instance._thread_manager.start()
        return _id
    
    def _enqueue(self, _id):
        _cls = type(self)
        if not _id in _cls._instance._thread_dict: raise ThreadException("Thread not registered")
        with _cls._instance._lock: _cls._instance._thread_queue.append(_id)

    def _wait(self, _id):
        _cls = type(self)
        while not _id in _cls._instance._return_dict: asyncio.run(asyncio.sleep(0.01))
        _ret = _cls._instance._return_dict.pop(_id)
        if isinstance(_ret, ThreadException): raise _ret
        return _ret

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
            cls._instance._lock = threading.Lock()
            cls._instance._thread_dict = {}
            cls._instance._running_thread_max = 16
            cls._instance._thread_queue = []
            cls._instance._running_threads = []
            cls._instance._return_dict = {}
            cls._instance._thread_manager = None
            # cls._instance._thread_manager = threading.Thread(target=cls._manager, args=(), daemon=True)
            # cls._instance._thread_manager.start()
        return cls._instance
    
    @classmethod
    def _manager(cls):
        while len(cls._instance._thread_dict) > 0:
            try:
                asyncio.run(asyncio.sleep(0.01))
                # Running threads only controled by the manager
                for _th_id in cls._instance._running_threads:
                    _th = cls._instance._thread_dict[_th_id]
                    if _th.is_alive(): continue
                    _ret = _th.join()
                    cls._instance._running_threads.pop(cls._instance._running_threads.index(_th_id))
                    cls._instance._return_dict[_th_id] = _ret
                    with cls._instance._lock: del cls._instance._thread_dict[_th_id]
                # Thread queue is controled by the manager and main process
                with cls._instance._lock:
                    if len(cls._instance._thread_queue) > 0 and len(cls._instance._running_threads) < cls._instance._running_thread_max:
                        _th_id = cls._instance._thread_queue.pop(0)
                    else: continue
                _th = _Thread_With_Return_Value(*cls._instance._thread_dict[_th_id])
                _th.start()
                cls._instance._thread_dict[_th_id] = _th
                cls._instance._running_threads.append(_th_id)
            except Exception as e: print(f"Thread manager raised an exception: {e}" + traceback.format_exc())

class Thread_With_Return_Value:
    def __init__(self, group=None, target=None, name=None, daemon=False, args=(), kwargs={}):
        self._thread_pool  = _Thread_Pool()
        self._id    : int  = self._thread_pool._register(group, target, name, daemon, args, kwargs)
        self._state : bool = False
    def start(self):
        if self._state: raise ThreadException("Thread already started")
        self._thread_pool._enqueue(self._id)
        self._state = True
    def join(self):
        if not self._state: raise ThreadException("Thread not started")
        _return = self._thread_pool._wait(self._id)
        self._state = False
        return _return
    
class ThreadException(Exception): pass
class _Thread_With_Return_Value(threading.Thread):
    def __init__(self, group=None, target=None, name=None, daemon=False, args=(), kwargs={}):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self._return = None
    def run(self):
        try:
            if self._target is not None: self._return = self._target(*self._args, **self._kwargs)
        except Exception as e: self._return = ThreadException(f"Thread raised an exception: {str(e)}\n" + traceback.format_exc())
    def join(self, *args):
        super().join(*args)
        return self._return