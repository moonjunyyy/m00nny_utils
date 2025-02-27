import time
from .escape_codes import EscapeCodes
from typing import Iterable, Generator, Any
from .formatting import seconds_to_human_readable, count_digits

class ProgressBar:
    def __init__(self,
                 iterable: Iterable,
                 length: int = None,
                 size: int = 50,
                 color: str = "g",
                 prefix: str = "",
                 descriptions: dict = None) -> None:
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
            self.current = i+1
            yield item
            self.print_progress()
        print()

    def set_descriptions(self, descriptions) -> None: self.descriptions = descriptions
    def set_prefix(self, prefix) -> None: self.prefix = prefix
    def generate_progress_bar(self, progress: float) -> str:
        full_blocks      = int(progress * self.size)
        fractional_index = int((progress * self.size - full_blocks) * (len(self.blocks) - 1))
        bar = self.blocks[-1] * full_blocks
        if fractional_index > 0 and full_blocks < self.size:
            bar += self.blocks[fractional_index]
        bar = bar.ljust(self.size)
        return bar
    
    def init_progress(self) -> None:
        # prefix current/total [progress bar] percentage [elapsed_time (+ elapsed_iter_time) / total_time / descriptions]
        print(f"{self.prefix}", end="")
        print(f"{self.current:>{count_digits(number=self.length)}}/{self.length:>{count_digits(number=self.length)}} [", end="")
        print(EscapeCodes.TERMINAL_CHAR_COLOR[self.color], end="")
        print(f"{self.generate_progress_bar(progress=0)}", end="")
        print(EscapeCodes.TERMINAL_RESET, end="")
        print(f"] 0.00%", end="")
        print(f" [00:00:00 (+ 00:00:00) / 00:00:00", end="")
        for key, value in self.descriptions.items():
            if "__" in key: print(f" / {value}", end="")
            else: print(f" / {key}: {value}", end="")
        print("]", end="", flush=True)

    def print_progress(self) -> None:
        current_time = time.time()
        # Make these variables can be accessed from outside for convenience.
        self.elapsed_time          = current_time - self.start_time
        self.elapsed_time_per_iter = current_time - self.last_time
        self.last_time             = current_time
        
        self.progress   = self.current / self.length
        self.remaining  = self.elapsed_time * (1 / self.progress - 1)
        self.total_time = self.elapsed_time + self.remaining
        bar = self.generate_progress_bar(progress=self.progress)

        # Clear the current line
        print(f"\r{EscapeCodes.TERMINAL_CLEAR_LINE}", end="")
        # prefix current/total [progress bar] percentage [elapsed_time (+ elapsed_iter_time) / total_time / descriptions]
        print(f"{self.prefix}", end="")
        print(f"{self.current:>{count_digits(number=self.length)}}/{self.length:>{count_digits(number=self.length)}} [", end="")
        print(f"{EscapeCodes.TERMINAL_CHAR_COLOR[self.color]}", end="")
        print(f"{bar}", end="")
        print(f"{EscapeCodes.TERMINAL_RESET}", end="")
        print(f"] {self.progress * 100:3.2f}% ", end="")
        print(f"[{seconds_to_human_readable(seconds=self.elapsed_time)} (+ {seconds_to_human_readable(seconds=self.elapsed_time_per_iter)}) / {seconds_to_human_readable(seconds=self.total_time)}", end="")
        for key, value in self.descriptions.items():
            if "__" in key: print(f" / {value}", end="")
            else: print(f" / {key}: {value}", end="")
        print("]", end="", flush=True)