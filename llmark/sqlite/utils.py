import sqlite3
from tqdm import tqdm
from typing import List, Tuple, Literal
from llmark.sqlite.float16 import FP16_PREFILL_BOUNDARY_KERNEL, FP16_DECODE_BOUNDARY_KERNEL
from llmark.sqlite.smooth_quant import SMOOTH_QUANT_PREFILL_BOUNDARY_KERNEL, SMOOTH_QUANT_DECODE_BOUNDARY_KERNEL

PREFILL_BOUNDARY_KERNEL = {
    "FP16": FP16_PREFILL_BOUNDARY_KERNEL,
    "smooth_quant": SMOOTH_QUANT_PREFILL_BOUNDARY_KERNEL
}

DECODE_BOUNDARY_KERNEL = {
    "FP16": FP16_DECODE_BOUNDARY_KERNEL,
    "smooth_quant": SMOOTH_QUANT_DECODE_BOUNDARY_KERNEL
}

class KernelFinder:
    _cursor: sqlite3.Cursor
    _first_prefill_kernel_time: Tuple[int, int]
    _prefill_kernel_times: List[Tuple[int, int]]
    prefill_start_kernel_id: int
    prefill_end_kernel_id: int
    decode_start_kernel_id: int
    decode_end_kernel_id: int
    quantization: str

    def __init__(self, cursor: sqlite3.Cursor, max_num_seqs: int, quantization: str) -> None:
        self._cursor = cursor

        QUERY = "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE '%PREFILL%' AND text NOT LIKE '%reqs: 0)%' AND text NOT LIKE '%reqs: 1)%' ORDER BY start ASC"
        if max_num_seqs == 1:
            QUERY = "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE '%PREFILL%' AND text NOT LIKE '%reqs: 0)%' ORDER BY start ASC LIMIT 100 OFFSET 1"
        self._cursor.execute(QUERY)

        nvtx_events = self._cursor.fetchall()
        self.quantization = quantization

        self._convert_nvtx_to_kernel_times(nvtx_events, quantization)
    
    def get_kernel_id_by_name(self, kernel_name: str):
        QUERY = f"SELECT id FROM StringIds WHERE value LIKE '%{kernel_name}%'"
        self._cursor.execute(QUERY)
        kernel_id = self._cursor.fetchone()
        assert kernel_id is not None, "Fucking Kernel id error"
        return kernel_id[0]
    
    def _convert_nvtx_to_kernel_times(self, nvtx_events: List[Tuple[int, int]], quantization: str):
        kernel_times = []

        boundary_kernel = PREFILL_BOUNDARY_KERNEL[quantization]

        self.prefill_start_kernel_id = self.get_kernel_id_by_name(boundary_kernel['start'])
        self.prefill_end_kernel_id = self.get_kernel_id_by_name(boundary_kernel['end'])
        
        for prefill in tqdm(nvtx_events):
            start_time, end_time = prefill
            i = 0
            j = -1
            QUERY = f"SELECT start, end, demangledName FROM KERNEL_WITH_INDEX WHERE correlationId IN (SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >= {start_time} AND end <= {end_time + 6000000}) ORDER BY start ASC"
            self._cursor.execute(QUERY)
            result = self._cursor.fetchall()
            is_error = False
            _kernel_start_time = None
            _kernel_end_time = None

            while True:
                _kernel = result[i]
                i += 1
                _kernel_start_time, _, _kernel_id = _kernel
                if _kernel_id == self.prefill_start_kernel_id:
                    break

            while True:
                try:
                    _kernel = result[j]
                except IndexError as e:
                    is_error = True
                    break
                j -= 1
                _, _kernel_end_time, _kernel_id = _kernel
                if _kernel_id == self.prefill_end_kernel_id:
                    break

            if _kernel_start_time is not None and _kernel_end_time is not None and not is_error:
                 kernel_times.append((_kernel_start_time, _kernel_end_time))

        self._first_prefill_kernel_time = kernel_times[0]
        self._prefill_kernel_times = kernel_times
    
    def get_prefill_boundary_kernel_ids(self):
        return self.prefill_start_kernel_id, self.prefill_end_kernel_id
    
    def get_decode_boundary_kernel_ids(self):
        boundary_kernel = DECODE_BOUNDARY_KERNEL[self.quantization]
        self.decode_start_kernel_id = self.get_kernel_id_by_name(boundary_kernel['start'])
        self.decode_end_kernel_id = self.get_kernel_id_by_name(boundary_kernel['end'])
        
        return self.decode_start_kernel_id, self.decode_end_kernel_id
    
    def get_start_kernels(self, start_kernel_id: int, end_kernel_id: int, phase: Literal['prefill', 'decode']):
        kernels = []

        if phase == 'decode':
            i = 0
            while i < len(self._prefill_kernel_times):
                _, prefill_end_time = self._prefill_kernel_times[i]

                if i + 1 < len(self._prefill_kernel_times):
                    prefill_start_time, _ = self._prefill_kernel_times[i+1]
                    QUERY = f"SELECT start, demangledName FROM KERNEL_WITH_INDEX WHERE start >= {prefill_end_time} AND end <= {prefill_start_time} ORDER BY start ASC"
                else:
                    QUERY = f"SELECT start, demangledName FROM KERNEL_WITH_INDEX WHERE start >= {prefill_end_time} ORDER BY start ASC"
                
                self._cursor.execute(QUERY)
                start_kernels = self._cursor.fetchall()

                temp = None
                is_continue = False
                j = 0
                for start_time, kernel_id in start_kernels:
                    j += 1
                    if kernel_id == start_kernel_id and not is_continue:
                        is_continue = True
                        temp = start_time
                        j = 0
                        continue

                    if kernel_id == end_kernel_id:
                        if j < 20:
                            temp = None
                        else:
                            kernels.append(temp)
                        is_continue = False
                        continue
                
                i += 1

        if phase == 'prefill':
            QUERY = f"SELECT start, demangledName FROM KERNEL_WITH_INDEX WHERE start >= {self._first_prefill_kernel_time[0]} ORDER BY start ASC"
            self._cursor.execute(QUERY)

            start_kernels = self._cursor.fetchall()

            for kernel in start_kernels:
                start_time = kernel[0]
                is_ok = False
                for prefill_kernel in self._prefill_kernel_times:
                    prefill_start_time, prefill_end_time = prefill_kernel
                    if prefill_start_time <= start_time and prefill_end_time >= start_time:
                        is_ok = prefill_start_time == start_time
                        break
        
                if is_ok:
                    kernels.append(start_time)

        return kernels