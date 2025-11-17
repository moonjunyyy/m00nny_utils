import requests
from .progress_bar import ProgressBar
from .formatting import bytes_to_human_readable
from typing import Generator


class DownloadURL(ProgressBar):
    def __init__(
        self,
        url: str,
        save_path: str,
        file_size: int = None,
        size: int = 50,
        color: str = "g",
        prefix: str = "",
        chunk_size: int = 2**22,  # 4 MiB
        descriptions: dict = {"__Speed": "0.00 MiB/s"},
    ) -> None:
        self.url = url
        self.save_path = save_path
        self.res = requests.get(url=url, stream=True)
        if self.res.status_code != 200:
            raise ConnectionError(f"Failed to connect to the server: {self.url}")
        self.dlen = int(self.res.headers.get("content-length", default=0))
        if self.dlen == 0:
            if file_size is not None:
                self.dlen = file_size
            else:
                raise ValueError(
                    f"Failed to get the content length of the file: {self.url}"
                )
        if self.dlen > chunk_size:
            self.__will_be_verbose = True
        else:
            self.__will_be_verbose = False
        self.n_iter = self.dlen // chunk_size + int(self.dlen % chunk_size != 0)
        self.chunk_size = chunk_size
        super().__init__(
            iterable=self.res.iter_content(chunk_size=chunk_size),
            length=self.n_iter,
            size=size,
            color=color,
            prefix=prefix,
            descriptions=descriptions,
        )

    def __verbose_worker(self) -> Generator:
        for _, chunk in enumerate(iterable=super().__iter__()):
            _len_downloaded = len(chunk)
            if self.elapsed_time_per_iter and self.elapsed_time_per_iter > 0:
                self.set_descriptions(
                    descriptions={
                        "__Speed": f"{bytes_to_human_readable(size=_len_downloaded / self.elapsed_time_per_iter)}/s"
                    }
                )
            yield chunk

    def __quiet_worker(self) -> None:
        ret = bytearray()
        for _, chunk in enumerate(
            iterable=self.res.iter_content(chunk_size=self.chunk_size)
        ):
            ret.extend(chunk)
        return ret

    def download(self) -> bytearray | None:
        if self.save_path is None:
            ret = bytearray()
            if self.__will_be_verbose:
                if self.__will_be_verbose:
                    for chunk in self.__verbose_worker():
                        ret.extend(chunk)
            else:
                ret.extend(self.__quiet_worker())
            return ret
        else:
            with open(file=self.save_path, mode="wb") as f:
                if self.__will_be_verbose:
                    for chunk in self.__verbose_worker():
                        f.write(chunk)
                else:
                    f.write(self.__quiet_worker())
