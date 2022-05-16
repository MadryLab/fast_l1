from typing import Callable, Dict, Optional
from numpy.lib.format import open_memmap
from pathlib import Path
import pandas as pd
import numpy as np

from multiprocessing import Pool

MAPPING_FIELD = 'index_mapping'


class Logger:
    def __init__(self,
                 logdir: str,
                 fields: Dict[str, type],
                 field_size: int,
                 cnk_size: int = 10_000) -> None:
        fields[MAPPING_FIELD] = np.int64
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
                                            mode='w+', dtype=fields[field],
                                            shape=(cnk_size, field_size))
            self.counters[field] = 0
            self.df.loc[i] = [field, 0, 0]

        self.metadata_path = Path(logdir) / 'metadata.csv'
        self.df = self.df.set_index('field')
        self.df.to_csv(self.metadata_path)

    def log_index_mapping(self, index_mapping):
        self.log(MAPPING_FIELD, index_mapping)

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


def agg(dat):
    i, mmap_name, reindexer_name, index, agg_fn = dat
    mmap = open_memmap(mmap_name, mode='r')
    reindexer = open_memmap(reindexer_name, mode='r')
    unindexer = np.argsort(reindexer[i])
    if not np.all(reindexer[i][unindexer] == np.arange(reindexer[i].shape[0])):
        return None
    new_row = mmap[i, unindexer]
    if index is not None:
        return new_row[index]

    if agg_fn:
        return agg_fn(new_row)
    return new_row


class Reader:
    def __init__(self, logdir) -> None:
        self.logdir = logdir
        self.df = pd.read_csv(Path(logdir) / 'metadata.csv')
        self.df = self.df.set_index('field')
        self.mmaps = {}
        for field_name, r in self.df.iterrows():
            self.mmaps[field_name] = []
            for chunk in range(r['max_chunk'] + 1):
                file_name = field_name + f'_{chunk}.npy'
                self.mmaps[field_name].append(Path(self.logdir) / file_name)

    def read_field(self, field_name, index=None,
                   agg_fn: Optional[Callable] = None):
        assert (index is None) or (agg_fn is None)

        zipped = list(zip(self.mmaps[field_name], self.mmaps[MAPPING_FIELD]))
        arrs_to_cat = []
        for j, (mmap_path, reindexer_path) in enumerate(zipped):
            p = Pool(10)
            map_args = [(i, mmap_path, reindexer_path, index, agg_fn)
                        for i in range(10_000)]
            rows = p.map(agg, map_args)
            first_none = rows.index(None)
            assert len(set(rows[first_none:])) == 1
            rows = rows[:first_none]
            assert not any(map(lambda x: x is None, rows))
            if index is not None or agg_fn is not None:
                arrs_to_cat.append(np.array(rows))
            else:
                arrs_to_cat.append(np.stack(rows, axis=0))

        arrs_to_cat[-1] = arrs_to_cat[-1]
        return np.concatenate(arrs_to_cat, axis=0)

    def fields(self):
        return self.mmaps.keys()
