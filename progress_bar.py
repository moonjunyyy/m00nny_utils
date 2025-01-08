import time
class ProgressBar:
    def __init__(self, iterable, size=30, descriptions={}):
        self.iterable = iterable
        self.length = len(iterable)
        self.start_time = time.time()
        self.last_time = self.start_time
        self.current = 0
        self.size = size
        self.blocks = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
        self.descriptions = descriptions

    def __iter__(self):
        for i, item in enumerate(self.iterable):
            self.current = i+1
            yield item
            self.print_progress()
        print()

    def count_digits(self, number):
        return len(str(number))

    def seconds_to_hms(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        f = seconds - int(seconds)
        hms = ""
        if h != 0: hms += f"{h:02}:"
        if hms != "" or m != 0: hms += f"{m:02}:"
        if hms != "" or s != 0: hms += f"{s:02}"
        if hms == "": hms = f"{int(f*1000):3d}ms"
        else: hms += f".{int(f*1000):03d}"
        return hms
    
    def set_descriptions(self, descriptions):
        self.descriptions = descriptions

    def print_progress(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        elapsed_iter = current_time - self.last_time
        self.last_time = current_time
        progress = self.current / self.length
        remaining = elapsed_time * (1 / progress - 1)
        total_time = elapsed_time + remaining

        full_blocks = int(progress * self.size)  # Full block count
        fractional_index = int((progress * self.size - full_blocks) * (len(self.blocks) - 1))

        bar = "█" * full_blocks  # Full blocks
        if fractional_index > 0 and full_blocks < self.size:
            bar += self.blocks[fractional_index]  # Add a fractional block
        bar = bar.ljust(self.size)  # Pad the remaining space with empty characters

        print(f"\033[K\r{self.current:>{self.count_digits(self.length)}}/{self.length:>{self.count_digits(self.length)}} [{bar}] {progress * 100:3.2f}%", end="")
        print(f" [{self.seconds_to_hms(elapsed_time)} (+ {self.seconds_to_hms(elapsed_iter)}) / {self.seconds_to_hms(total_time)}]", end="")
        for key, value in self.descriptions.items():
            print(f" - {key}: {value}", end="")