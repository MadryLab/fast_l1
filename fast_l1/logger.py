from typing import Callable, Dict, Optional, Tuple
from numpy.lib.format import open_memmap
from pathlib import Path
import pandas as pd
import numpy as np

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
        self.df = pd.DataFrame(columns=['field', 'max_chunk', 'counter'])

        for i, field in enumerate(fields):
            self.chunks[field] = 0
            file_name = field + f'_{self.chunks[field]}.npy'
            self.mmaps[field] = open_memmap(Path(logdir) / file_name,
                                            mode='w+', dtype=fields[field][0],
                                            shape=(cnk_size, fields[field][1]))
            self.counters[field] = 0
            self.df.loc[i] = [field, 0, 0]

        self.metadata_path = Path(logdir) / 'metadata.csv'
        self.df = self.df.set_index('field')
        self.df.to_csv(self.metadata_path)

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
        
        self.df.loc[field_name] = [self.chunks[field_name],
                                   self.counters[field_name]]
        self.df.to_csv(self.metadata_path)

    def flush(self):
        for field_name in self.mmaps:
            self.mmaps[field_name].flush()


class Reader:
    def __init__(self, logdir) -> None:
        self.logdir = logdir
        self.df = pd.read_csv(Path(logdir) / 'metadata.csv')
        self.mmaps = {}
        for field, r in self.df.iterrows():
            self.mmaps[field] = []
            for chunk in range(r['max_chunk']):
                file_name = field + f'_{chunk}.npy'
                self.mmaps[field].append(open_memmap(
                    Path(self.logdir) / file_name, mode='r'))

    def read_field(self, field_name, index=None, agg_fn: Optional[Callable]=None):
        assert (index is None) or (agg_fn is None)

        def agg(x):
            return x[:, index] if index is not None else \
                   (agg_fn(x) if agg_fn is not None else x)

        arrs_to_cat = [agg(m) for m in self.mmaps[field_name]]
        arrs_to_cat[-1] = arrs_to_cat[-1][:self.df.loc[field_name]['counter']]
        return np.concatenate(arrs_to_cat, axis=0)

    def fields(self):
        return self.mmaps.keys()
