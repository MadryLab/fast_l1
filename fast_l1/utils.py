from ffcv.pipeline.operation import Operation, AllocationQuery
from ffcv.pipeline.compiler import Compiler
from dataclasses import replace
import numpy as np


class ReorderOp(Operation):
    def __init__(self, num_outputs) -> None:
        super().__init__()
        self.num_keep = num_outputs
        self.index_mapping = np.arange(num_outputs)

    def generate_code(self):
        mapping = self.index_mapping
        num_keep = self.num_keep
        parallel_range = Compiler.get_iterator()
        print(num_keep)

        def remap(inp, dst):
            for i in parallel_range(num_keep):
                dst[:, i] = inp[:, mapping[i]]
            return dst

        remap.is_parallel = True
        return remap

    def declare_state_and_memory(self, previous_state):
        return 
               
