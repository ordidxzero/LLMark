import sqlite3, re
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm

@dataclass
class Kernel:
    start: int
    end: int
    id: int
    name: str

class StateMachine(ABC):
    @abstractmethod
    def transform(self, kernel: Kernel) -> Optional[str | Tuple[str, Kernel]]:
        raise Exception("Not Implemented!")

    @abstractmethod
    def log(self, alias: str, kernel: Kernel):
        raise Exception("Not Implemented!")
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, List[float]]:
        raise Exception("Not Implemented!")
    
    @abstractmethod
    def reset(self) -> None:
        raise Exception("Not Implemented!")

def get_prefill_events(cursor: sqlite3.Cursor, max_num_seqs: int, num_prefill: int) -> List[Tuple[int, int]]:
    QUERY = "SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE '%PREFILL%' AND text NOT LIKE '%reqs: 0)%' AND text NOT LIKE '%reqs: 1)%' ORDER BY start ASC"
    if max_num_seqs == 1:
        QUERY = "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE '%PREFILL%' AND text NOT LIKE '%reqs: 0)%' ORDER BY start ASC LIMIT 100 OFFSET 1"
    cursor.execute(QUERY)

    result = cursor.fetchall()

    if max_num_seqs == 1:
        return result[-num_prefill:]

    output = []
    for event in result:
        start, end, text = event
        match = re.search(r'\(#reqs: (\d+)\)', text)
        assert match is not None, "Error"
        num_reqs = int(match.group(1))

        if max_num_seqs == 256:
            if num_reqs == 105 or num_reqs > 60:
                output.append((start, end))
        else:
            if num_reqs > max_num_seqs - 2:
                output.append((start, end))

    return output

def get_prefill_kernels(cursor: sqlite3.Cursor, quantization: str, prefill_events: List[Tuple[int, int]]) -> List[List[Kernel]]:
    output: List[List[Kernel]] = []
    for prefill_event in tqdm(prefill_events, desc="get_prefill_kernels"):
        start_time, end_time = prefill_event
        end_time_expand = 15000000 if quantization == 'smooth_quant' else 6000000
        QUERY = f"SELECT kernel.start, kernel.end, s.value, kernel.demangledName FROM StringIds AS s JOIN KERNEL_WITH_INDEX AS kernel ON s.id = kernel.demangledName WHERE kernel.correlationId IN (SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >= {start_time} AND end <= {end_time + end_time_expand})"
        cursor.execute(QUERY)
        result = cursor.fetchall()
        kernels: List[Kernel] = []

        for pk in result:
            start, end, name, id = pk
            kernels.append(Kernel(start, end, id, name))

        output.append(kernels)
        

    return output

def get_decode_kernels(cursor: sqlite3.Cursor, offset: int) -> List[Kernel]:
    QUERY = f"SELECT kernel.start, kernel.end, s.value, kernel.demangledName FROM StringIds AS s JOIN KERNEL_WITH_INDEX AS kernel ON s.id = kernel.demangledName WHERE kernel.correlationId IN (SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >= {offset}) ORDER BY start ASC"
    cursor.execute(QUERY)
    result = cursor.fetchall()
    kernels = []

    enable_skip = True
    for dk in result:
        start, end, name, id = dk
        if 'indexSelect' in name:
            enable_skip = False
        if enable_skip:
            continue
        kernels.append(Kernel(start, end, id, name))

    return kernels

def find_invalid_kernels(kernels: List[Kernel], decode_boundary: Tuple[Kernel, Kernel]) -> Tuple[int, List[Kernel]]:
    _, end_kernel = decode_boundary
    idx = 0
    is_invalid_kernel = False
    
    for kernel in kernels:
        idx += 1
        if kernel.id == end_kernel.id:
            is_invalid_kernel = idx <= 100
            break

    return is_invalid_kernel

def adjust_decode_kernels(kernels: List[Kernel], decode_boundary: Tuple[Kernel, Kernel]) -> Tuple[int, List[Kernel]]:
    start_kernel, end_kernel = decode_boundary
    idx = 0
    is_valid_kernels = None
    result = []
    
    for kernel in kernels:
        idx += 1
        if kernel.id == start_kernel.id and is_valid_kernels == False:
            is_valid_kernels = True
        if is_valid_kernels:
            result.append(kernel)
        if kernel.id == end_kernel.id and is_valid_kernels == None:
            is_valid_kernels = False
        if kernel.id == end_kernel.id and is_valid_kernels == True:
            break

    return idx, result