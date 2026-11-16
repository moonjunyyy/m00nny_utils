import shutil
import time
from typing import Iterable, Generator
from .escape_codes import EscapeCodes
from .formatting import seconds_to_human_readable, count_digits


class ProgressBar:
    min_size = 10  # the bar is not drawn shorter than this when the terminal is narrow

    def __init__(
        self,
        iterable: Iterable,
        length: int = None,
        size: int = 50,
        color: str = "g",
        prefix: str = "",
        descriptions: dict = None,
    ) -> None:
        self.blocks = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]

        self.iterable = iterable
        self.length = len(iterable) if length is None else length
        self.start_time = time.time()
        self.last_time = self.start_time
        self.current = 0
        self.size = size
        self.color = color
        self.prefix = prefix
        self.descriptions = {} if descriptions is None else descriptions

        self.elapsed_time = None
        self.elapsed_time_per_iter = None
        self.progress = None
        self.remaining = None
        self.total_time = None

    def __iter__(self) -> Generator:
        self.init_progress()
        for i, item in enumerate(iterable=self.iterable):
            self.current = i + 1
            yield item
            self.print_progress()
        print()

    def set_descriptions(self, descriptions) -> None:
        self.descriptions = descriptions

    def set_prefix(self, prefix) -> None:
        self.prefix = prefix

    def generate_progress_bar(self, progress: float, size: int = None) -> str:
        size = self.size if size is None else size
        full_blocks = int(progress * size)
        fractional_index = int(
            (progress * size - full_blocks) * (len(self.blocks) - 1)
        )
        bar = self.blocks[-1] * full_blocks
        if fractional_index > 0 and full_blocks < size:
            bar += self.blocks[fractional_index]
        bar = bar.ljust(size)
        return bar

    def fit(self, head: str, tail: str) -> tuple:
        # The line is drawn on one row of the terminal. A longer one wraps, and the carriage return
        # then comes back to its last row only: every update would leave a row behind.
        # The bar gives up its width first (down to min_size), then the tail is cut.
        columns = shutil.get_terminal_size(fallback=(120, 24)).columns - 1
        size = max(min(self.size, columns - len(head) - len(tail)), self.min_size)
        room = max(columns - len(head) - size, 0)
        if len(tail) > room:
            tail = tail[: max(room - 1, 0)] + "…" if room > 0 else ""
        return size, tail

    def descriptions_text(self) -> str:
        text = ""
        for key, value in self.descriptions.items():
            if "__" in key:
                text += f" / {value}"
            else:
                text += f" / {key}: {value}"
        return text

    def init_progress(self) -> None:
        # prefix current/total [progress bar] percentage [elapsed_time (+ elapsed_iter_time) / total_time / descriptions]
        head = (
            f"{self.prefix} " +
            f"{self.current:>{count_digits(number=self.length)}}/" +
            f"{self.length:>{count_digits(number=self.length)}} ["
        )
        tail = "] 0.00% [00:00:00 (+ 00:00:00) / 00:00:00" + self.descriptions_text() + "]"
        size, tail = self.fit(head, tail)
        print(
            head +
            f"{EscapeCodes.TERMINAL_CHAR_COLOR[self.color]}" +
            f"{self.generate_progress_bar(progress=0, size=size)}" +
            f"{EscapeCodes.TERMINAL_RESET}" +
            tail,
            end="", flush=True,
        )

    def print_progress(self) -> None:
        current_time = time.time()
        # Make these variables can be accessed from outside for convenience.
        self.elapsed_time = current_time - self.start_time
        self.elapsed_time_per_iter = current_time - self.last_time
        self.last_time = current_time

        self.progress = self.current / self.length
        self.remaining = self.elapsed_time * (1 / self.progress - 1)
        self.total_time = self.elapsed_time + self.remaining

        head = (
            f"{self.prefix} " +
            f"{self.current:>{count_digits(number=self.length)}}/" +
            f"{self.length:>{count_digits(number=self.length)}} ["
        )
        tail = (
            f"] {self.progress * 100:3.2f}% " +
            f"[{seconds_to_human_readable(seconds=self.elapsed_time)} " +
            f"(+ {seconds_to_human_readable(seconds=self.elapsed_time_per_iter)}) " +
            self.descriptions_text() + "]"
        )
        size, tail = self.fit(head, tail)
        print(
            f"\r{EscapeCodes.TERMINAL_CLEAR_LINE}" +
            head +
            f"{EscapeCodes.TERMINAL_CHAR_COLOR[self.color]}" +
            f"{self.generate_progress_bar(progress=self.progress, size=size)}" +
            f"{EscapeCodes.TERMINAL_RESET}" +
            tail,
            end="", flush=True,
        )
