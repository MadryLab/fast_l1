from typing import Dict, Tuple
from numpy.lib.format import open_memmap
from pathlib import Path


class Logger:
    def __init__(self,
                 logdir: str,
                 fields: Dict[str, Tuple],
                 cnk_size: int = 10_000) -> None:
        self.logdir = logdir
        self.mmaps = {}
        self.counters = {}
        self.chunks = {}
        self.chunk_size = cnk_size

        for field in fields:
            self.chunks[field] = 0
            file_name = field + f'_{self.chunks[field]}.npy'
            self.mmaps[field] = open_memmap(Path(logdir) / file_name,
                                            mode='w+', dtype=fields[field][0],
                                            shape=(cnk_size, fields[field][1]))
            self.counters[field] = 0

    def log(self, field_name: str, field_value):
        mmap = self.mmaps[field_name]
        mmap[self.counters[field_name]] = field_value
        self.counters[field_name] += 1

        if self.counters[field_name] == self.chunk_size:
            self.counters[field_name] = 0
            self.chunks[field_name] += 1
            file_name = field_name + f'_{self.chunks[field_name]}.npy'
            self.mmaps[field_name].flush()
            self.mmaps[field_name] = open_memmap(Path(self.logdir) / file_name,
                                                 mode='w+', dtype=mmap.dtype,
                                                 shape=mmap.shape)

    def flush(self):
        for field_name in self.mmaps:
            self.mmaps[field_name].flush()


class Reader:
    def __init__(self) -> None:
        pass

    def read():
        pass