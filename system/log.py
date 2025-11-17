isdmport os
import sys
import logging
from typing import Union, Self
from logging import StreamHandler, FileHandler
from logging.handlers import QueueListener, QueueHandler
import multiprocessing as mp


class _Global_Log_Queue_Listener:
    __instance = None
    __initialized = False

    def __new__(cls, *args, **kwargs) -> Self:
        if cls.__instance is None:
            cls.__instance = super(_Global_Log_Queue_Listener, cls).__new__(cls)
            cls.__instance.GLOBAL_LISTENNER = None
            cls.__instance.GLOBAL_LOG_QUEUE = mp.Queue(maxsize=-1)
            cls.__instance.GLOBAL_LISTENING = False
            cls.__instance.cnt = 0
        return cls.__instance

    def _init_log_queue(
        self,
        level=logging.INFO,
        path: os.PathLike = None,
        use_STDOUT: bool = True,
        use_STDERR: bool = False,
    ) -> QueueListener:
        try:
            if self.GLOBAL_LISTENING:
                if self.GLOBAL_LISTENNER is not None:
                    raise Exception(
                        "Global Listener already running. Did you forget to stop?"
                    )
            handlers = []
            if use_STDOUT:
                stream_handler = StreamHandler(stream=sys.stdout)
                stream_handler.setFormatter(
                    fmt=logging.Formatter(
                        fmt="[ %(asctime)s | %(name)s | %(levelname)s ]: %(message)s"
                    )
                )
                stream_handler.setLevel(level=level)
                handlers.append(stream_handler)
            if use_STDERR:
                error_handler = StreamHandler(stream=sys.stderr)
                error_handler.setFormatter(
                    fmt=logging.Formatter(
                        fmt="[ %(asctime)s | %(name)s | %(levelname)s ]: %(message)s"
                    )
                )
                error_handler.setLevel(level=logging.WARNING)
                handlers.append(error_handler)
            if path is not None:
                file_handeler = FileHandler(filename=f"{path}/global.log")
                file_handeler.setFormatter(
                    fmt=logging.Formatter(
                        fmt="[ %(asctime)s | %(name)s | %(levelname)s ]: %(message)s"
                    )
                )
                file_handeler.setLevel(level=logging.DEBUG)
                handlers.append(file_handeler)
            self.GLOBAL_LISTENNER = logging.handlers.QueueListener(
                self.GLOBAL_LOG_QUEUE, *handlers, respect_handler_level=True
            )
            self.GLOBAL_LISTENNER.start()
            self.GLOBAL_LISTENING = True
            self.cnt += 1
        except Exception as e:
            print(f"Error: {e}")
            return None

    def __init__(
        self,
        level=logging.INFO,
        path: os.PathLike = None,
        use_STDOUT: bool = True,
        use_STDERR: bool = False,
    ) -> None:
        if self.__initialized:
            return
        self.GLOBAL_LISTENING = False
        self.GLOBAL_LISTENNER = None
        self.GLOBAL_LOG_QUEUE = mp.Queue(maxsize=-1)
        self._init_log_queue(
            level=level, path=path, use_STDOUT=use_STDOUT, use_STDERR=use_STDERR
        )
        self.__initialized = True
        return

    def get_log_queue(self) -> mp.Queue:
        return self.GLOBAL_LOG_QUEUE

    def stop(self) -> None:
        if self.GLOBAL_LISTENNING:
            self.GLOBAL_LISTENNER.stop()
            self.GLOBAL_LISTENING = False
            self.GLOBAL_LISTENNER = None
            self.cnt -= 1
        else:
            print("ERROR: Listener is not running!")

    def __del__(self) -> None:
        if self.cnt == 0:
            self.stop()
            del self.__instance


class Log:
    def get_level(self, level: str) -> int:
        if isinstance(level, int):
            return level
        if isinstance(level, str):
            if level.upper() == "DEBUG":
                return logging.DEBUG
            if level.upper() == "INFO":
                return logging.INFO
            if level.upper() == "WARNING":
                return logging.WARNING
            if level.upper() == "ERROR":
                return logging.ERROR
            if level.upper() == "CRITICAL":
                return logging.CRITICAL
        return logging.INFO

    def __init__(
        self,
        name: Union[str, None] = None,
        level: Union[int, str] = "DEBUG",
        global_level: Union[int, str] = "INFO",
        path: os.PathLike = None,
        use_STDOUT: bool = True,
        use_STDERR: bool = False,
    ) -> None:
        level = self.get_level(level=level)
        global_level = self.get_level(level=global_level)

        self.listener = _Global_Log_Queue_Listener(
            level=global_level, path=path, use_STDOUT=use_STDOUT, use_STDERR=use_STDERR
        )
        self.level = level
        self.logger = self.get_logger(name=name, level=level)

    def get_logger(
        self, name: Union[str, None] = None, level: Union[int, str] = "INFO"
    ) -> logging.Logger:
        logger = logging.getLogger(name=name)
        logger.setLevel(level=self.get_level(level=level))
        handler = QueueHandler(queue=self.listener.get_log_queue())
        handler.setLevel(level=self.get_level(level=level))
        logger.addHandler(hdlr=handler)
        return logger

    def log(self, message: str) -> None:
        self.logger.log(level=self.level, msg=message)

    def debug(self, message: str) -> None:
        self.logger.debug(msg=message)

    def info(self, message: str) -> None:
        self.logger.info(msg=message)

    def warning(self, message: str) -> None:
        self.logger.warning(msg=message)

    def error(self, message: str) -> None:
        self.logger.error(msg=message)

    def critical(self, message: str) -> None:
        self.logger.critical(msg=message)

    def stdout(self, message: str) -> None:
        print(message, file=self.logger)

    def stderr(self, message: str) -> None:
        print(message, file=sys.stderr)
