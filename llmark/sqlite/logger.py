import sqlite3
from typing import List
from llmark.sqlite import Kernel, get_decode_kernels

class KernelLogger:
    cursor: sqlite3.Cursor
    prefill_kernels: List[List[Kernel]]
    last_prefill: List[Kernel]

    def __init__(self, cursor: sqlite3.Cursor, prefill_kernels: List[List[Kernel]]):
        self.cursor = cursor

        self.prefill_kernels = prefill_kernels
        self.last_prefill = prefill_kernels[-1]

    def _fprint(self, i: int, kernel: Kernel):
        print(f"{i}".ljust(4), ":", kernel.start, kernel.end, kernel.name)

    def _iter_print(self, title: str, kernels: List[Kernel]):
        print("=" * 10, title, "=" * 10)
        for i, kernel in enumerate(kernels[:70]):
            self._fprint(i, kernel)
        print('--- skip ---')
        for i, kernel in enumerate(kernels[-70:]):
            self._fprint(i, kernel)

    def print_prefill_kernels(self):
        self._iter_print('Prefill', self.last_prefill)

    def print_decode_kernels(self):
        kernels = get_decode_kernels(self.cursor, self.last_prefill[-1].end)
        is_invalid_kernel = True
        result = []
        for k in kernels:
            result.append(k)
            if 'void at::native::reduce_kernel' in k.name:
                if is_invalid_kernel:
                    is_invalid_kernel = False
                else:
                    break

        self._iter_print('Decode', result)
