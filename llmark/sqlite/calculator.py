import sqlite3
from typing import List, Tuple, Literal, Set, Dict
from tqdm import tqdm
from collections import defaultdict
from llmark.sqlite import get_decode_kernels, adjust_decode_kernels, find_invalid_kernels, Kernel, StateMachine

class KernelCalculator:
    cursor: sqlite3.Cursor
    connection: sqlite3.Connection
    prefill_kernels: List[List[Kernel]]
    decode_kernels: List[Kernel]
    decode_boundary: Tuple[Kernel, Kernel]
    prefill_boundary: Tuple[Kernel, Kernel]
    workload: Literal['prefill_heavy', 'decode_heavy']

    def __init__(self, cursor: sqlite3.Cursor, connection: sqlite3.Connection, prefill_kernels: List[List[Kernel]], workload: Literal['prefill_heavy', 'decode_heavy']):
        self.cursor = cursor
        self.connection = connection
        self.workload = workload
        self.prefill_kernels = prefill_kernels
        self.prefill_boundary = (prefill_kernels[-1][0], prefill_kernels[-1][-1])

        decode_kernels = get_decode_kernels(cursor, self.prefill_boundary[-1].end)
        result = []
        for k in decode_kernels:
            if len(result) == 0:
                result.append(k)
            if 'void at::native::reduce_kernel' in k.name:
                    result.append(k)
                    break
        self.decode_kernels = decode_kernels
        self.decode_boundary = (result[0], result[-1])
    
    def print_metrics(self, metrics):
        print("LATENCY                  : ", round(sum(metrics['durations']) / len(metrics['durations']), 5))
        print("QKV PROJ                 : ", round((sum(metrics['qkv_proj']) / len(metrics['qkv_proj'])) / 1000000, 5))
        print("SELF ATTN                : ", round((sum(metrics['self_attn']) / len(metrics['self_attn'])) / 1000000, 5))
        print("O PROJ                   : ", round((sum(metrics['o_proj']) / len(metrics['o_proj'])) / 1000000, 5))
        print("GATE UP PROJ             : ", round((sum(metrics['gate_up_proj']) / len(metrics['gate_up_proj'])) / 1000000, 5))
        print("DOWM PROJ                : ", round((sum(metrics['down_proj']) / len(metrics['down_proj'])) / 1000000, 5))
        print("LM HEAD                  : ", round((sum(metrics['lm_head']) / len(metrics['lm_head'])) / 1000000, 5))
        if 'qkv_quant' in metrics:
            print("QUANT OVERHEAD (qkv)     : ", round((sum(metrics['qkv_quant']) / len(metrics['qkv_quant'])) / 1000000, 5))
            print("QUANT OVERHEAD (o)       : ", round((sum(metrics['o_quant']) / len(metrics['o_quant'])) / 1000000, 5))
            print("QUANT OVERHEAD (gate_up) : ", round((sum(metrics['gate_up_quant']) / len(metrics['gate_up_quant'])) / 1000000, 5))
            print("QUANT OVERHEAD (down)    : ", round((sum(metrics['down_quant']) / len(metrics['down_quant'])) / 1000000, 5))

    def calc_prefill_kernels(self, state_machine: StateMachine):
        metrics = defaultdict(list)

        for p_kernels in self.prefill_kernels:
            for kernel in p_kernels:
                name = state_machine.transform(kernel)
                if name is not None:
                    if type(name) == tuple:
                        name, kernel = name
                    state_machine.log(name, kernel) # type: ignore
            _metrics = state_machine.get_metrics()

            for k, v in _metrics.items():
                if k == 'lm_head':
                    metrics[k].append(sum(v) / len(v))
                else:
                    metrics[k].append(sum(v))

            start_time = p_kernels[0].start
            end_time = p_kernels[-1].end
            metrics['durations'].append(round((end_time - start_time) / 1000000, 2))
            state_machine.reset()
        
        self.print_metrics(metrics)

    def calc_decode_kernels(self, state_machine: StateMachine):
        is_init = True
        idx = 0
        _idx = 0
        pbar = tqdm(total=2048 if self.workload == 'decode_heavy' else 128)
        metrics = defaultdict(list)

        while True:
            idx += _idx
            _idx = 0
            kernels: List[Kernel] = []
            if is_init:
                _idx, adjusted = adjust_decode_kernels(self.decode_kernels, self.decode_boundary)
                idx += _idx
                kernels.extend(adjusted)
                is_init = False
                _idx = 0
            else:
                if len(self.decode_kernels[idx:]) == 0:
                    break

                is_start = False
                for kernel in self.decode_kernels[idx:]:
                    _idx += 1
                    if not is_start and kernel.id == self.decode_boundary[0].id:
                        is_start = True

                    if is_start:
                        kernels.append(kernel)

                    if is_start and kernel.id == self.decode_boundary[1].id:
                        break
        
            
            for kernel in kernels:
                name = state_machine.transform(kernel)

                if name is not None:
                    if type(name) == tuple:
                        name, kernel = name
                    state_machine.log(name, kernel) # type: ignore

            _metrics = state_machine.get_metrics()

            for k, v in _metrics.items():
                if k == 'lm_head':
                    metrics[k].append(sum(v) / len(v))
                else:
                    metrics[k].append(sum(v))

            pbar.update(1)
            if len(kernels) == 0:
                continue
            start_time = kernels[0].start
            end_time = kernels[-1].end
            metrics['durations'].append(round((end_time - start_time) / 1000000, 2))
            state_machine.reset()
        breakpoint()
        self.print_metrics(metrics)

    def _validate_dram_bandwidth(self, names: Set[str], bandwidth: Dict[str, List[float]]):
        is_valid_num_key = len(names) == len(bandwidth)
        is_valid_num_bandwidth = []
        values = []

        for k, v in bandwidth.items():
            if k == 'lm_head':
                is_valid_num_bandwidth.append(len(v) == 1)
                values.append(v[0])
                continue

            is_valid_num_bandwidth.append(len(v) == 32)
            values.append(sum(v) / len(v))

        is_valid_value = list(map(lambda x: x == values[0], values))

        return is_valid_num_key and all(is_valid_num_bandwidth) and not all(is_valid_value)
    
    def calc_gpu_utils(self, HW_OPS: int, SW_OPS: int, time: float):
        _NANO_SECONDS = 1_000_000_000
        ACTUAL_OPS = SW_OPS * _NANO_SECONDS / time
        return round((ACTUAL_OPS / HW_OPS) * 100, 3)

    def calc_dram_bandwidth(self, phase: str, state_machine: StateMachine, max_num_seqs: int, quantization:str):
        _HW_OPS = 624_000_000_000_000 if quantization == 'smooth_quant' else 312_000_000_000_000
        _BYTES = 1 if quantization == 'smooth_quant' else 2
        _SW_OPS = {
            "qkv_proj": 2 * 4096 * 4096 * max_num_seqs + 2 * 2 * 4096 * 128 * 8 * max_num_seqs,
            "o_proj": 2 * 4096 * 4096 * max_num_seqs,
            "self_attn": 4 * 4096 * 8 * 1024 * max_num_seqs,
            "gate_up_proj": 2 * 2 * 4096 * 14336 * max_num_seqs + 14336 * 14336,
            "down_proj": 2 * 4096 * 14336 * max_num_seqs,
            "lm_head": 2 * 4096 * 128256 * max_num_seqs
        }
        _PARAMS = {
            "qkv_proj": (4096 * 4096 + 2 * 4096 * 128 * 8) * _BYTES,
            "self_attn": 0,
            "o_proj": 4096 * 4096 * _BYTES,
            "gate_up_proj": 2 * 4096 * 14336 * _BYTES,
            "down_proj": 4096 * 14336 * _BYTES,
            "lm_head": 4096 * 128256 * _BYTES
        }
        is_init = phase == 'decode'
        idx = 0
        _idx = 0
        dram_bandwidth = defaultdict(list)
        sm_active = defaultdict(list)
        compute_warp = defaultdict(list)
        unallocated_warp = defaultdict(list)
        tensor_active = defaultdict(list)
        avg_computation_time = defaultdict(list)
        result = defaultdict(list)
        sm_result = defaultdict(list)
        compute_warp_result = defaultdict(list)
        unallocated_warp_result = defaultdict(list)
        tensor_active_result = defaultdict(list)
        names: Set[str] = set()
        _kernels = self.decode_kernels if phase == 'decode' else [_k for k in self.prefill_kernels for _k in k]
        _boundary = self.decode_boundary if phase == 'decode' else self.prefill_boundary
        pbar = tqdm(total=len(_kernels))


        while True:
            idx+= _idx
            _idx = 0
            kernels: List[Kernel] = []
            if is_init:
                _idx, adjusted = adjust_decode_kernels(_kernels, _boundary)
                idx += _idx
                kernels.extend(adjusted)
                is_init = False
                pbar.update(_idx)
            else:
                if len(_kernels[idx:]) == 0:
                    break

                is_start = False
                for kernel in _kernels[idx:]:
                    _idx += 1
                    pbar.update(1)
                    if not is_start and kernel.id == _boundary[0].id:
                        is_start = True

                    if is_start:
                        kernels.append(kernel)

                    if is_start and kernel.id == _boundary[1].id:
                        break

            for kernel in kernels:
                name = state_machine.transform(kernel)

                if name is not None:
                    if type(name) == tuple:
                        name, kernel = name

                    if ('quant' in name) or ('reshape_and_cache_flash_kernel' in kernel.name) or ('combine_kernel' in kernel.name):
                        continue
                    
                    start_bucket = int(kernel.end / 1000000)
                    QUERY = f"SELECT ROUND(AVG(value), 3) FROM DRAM_BANDWIDTH_SEG INDEXED BY idx_bdw_bucket WHERE seg_start_bucket BETWEEN ? AND ? AND start <= ? AND end >= ?"
                    self.cursor.execute(QUERY, (start_bucket - 5,start_bucket, kernel.end, kernel.start))

                    avg_throughput = self.cursor.fetchone()[0]

                    QUERY = f"SELECT ROUND(AVG(value), 3) FROM SM_ACTIVE_SEG INDEXED BY idx_sm_bucket WHERE seg_start_bucket BETWEEN ? AND ? AND start <= ? AND end >= ?"
                    self.cursor.execute(QUERY, (start_bucket - 5,start_bucket, kernel.end, kernel.start))

                    avg_sm_active = self.cursor.fetchone()[0]

                    QUERY = f"SELECT ROUND(AVG(value), 3) FROM COMPUTE_WARP_SEG INDEXED BY idx_compute_warp_bucket WHERE seg_start_bucket BETWEEN ? AND ? AND start <= ? AND end >= ?"
                    self.cursor.execute(QUERY, (start_bucket - 5,start_bucket, kernel.end, kernel.start))

                    avg_compute_warp = self.cursor.fetchone()[0]

                    QUERY = f"SELECT ROUND(AVG(value), 3) FROM UNALLOCATED_WARP_SEG INDEXED BY idx_unallocated_warp_bucket WHERE seg_start_bucket BETWEEN ? AND ? AND start <= ? AND end >= ?"
                    self.cursor.execute(QUERY, (start_bucket - 5,start_bucket, kernel.end, kernel.start))

                    avg_unallocated_warp = self.cursor.fetchone()[0]

                    QUERY = f"SELECT ROUND(AVG(value), 3) FROM TENSOR_ACTIVE_SEG INDEXED BY idx_tensor_bucket WHERE seg_start_bucket BETWEEN ? AND ? AND start <= ? AND end >= ?"
                    self.cursor.execute(QUERY, (start_bucket - 5,start_bucket, kernel.end, kernel.start))

                    avg_tensor_active = self.cursor.fetchone()[0]

                    if avg_throughput is not None and avg_sm_active is not None and avg_compute_warp is not None and avg_unallocated_warp is not None and avg_tensor_active is not None:
                        if name == 'self_attn_end':
                            name = 'self_attn'
                        names.add(name) # type: ignore
                        avg_computation_time[name].append(kernel.end - kernel.start)
                        dram_bandwidth[name].append(avg_throughput)
                        sm_active[name].append(avg_sm_active)
                        compute_warp[name].append(avg_compute_warp)
                        unallocated_warp[name].append(avg_unallocated_warp)
                        tensor_active[name].append(avg_tensor_active)
                    else:
                        breakpoint()
                        break

            if self._validate_dram_bandwidth(names, dram_bandwidth):
                for k, v in dram_bandwidth.items():
                    if k == 'lm_head':
                        result[k].append(v[0])
                    else:
                        result[k].append(sum(v) / len(v))

                for k, v in sm_active.items():
                    if k == 'lm_head':
                        sm_result[k].append(v[0])
                    else:
                        sm_result[k].append(sum(v) / len(v))

                for k, v in compute_warp.items():
                    if k == 'lm_head':
                        compute_warp_result[k].append(v[0])
                    else:
                        compute_warp_result[k].append(sum(v) / len(v))

                for k, v in unallocated_warp.items():
                    if k == 'lm_head':
                        unallocated_warp_result[k].append(v[0])
                    else:
                        unallocated_warp_result[k].append(sum(v) / len(v))

                for k, v in tensor_active.items():
                    if k == 'lm_head':
                        tensor_active_result[k].append(v[0])
                    else:
                        tensor_active_result[k].append(sum(v) / len(v))
                pbar.update(1)
            
            dram_bandwidth = defaultdict(list)   
            sm_active = defaultdict(list)
            compute_warp = defaultdict(list)
            unallocated_warp = defaultdict(list)
            tensor_active = defaultdict(list)
            names = set()
            state_machine.reset()

        if len(result) > 0:
            print("=== DRAM Bandwidth (Throughput %) ===")
            for k, v in result.items():
                actual_bdw = 1935 * (sum(v) / len(v)) / 100
                print(f"{k.upper().ljust(15)}: {sum(v) / len(v):.3f}% ({actual_bdw:.3f} GB/s, Load latency: {(_PARAMS[k] / actual_bdw) / 1000:.3f} μs)")
            print("=== SM Active (Throughput %) ===")
            for k, v in sm_result.items():
                print(f"{k.upper().ljust(15)}: {sum(v) / len(v):.3f}%")
            print("=== Compute Warps in Flight (Throughput %) ===")
            for k, v in compute_warp_result.items():
                print(f"{k.upper().ljust(15)}: {sum(v) / len(v):.3f}%")
            print("=== Unallocated Warps in Active SMs (Throughput %) ===")
            for k, v in unallocated_warp_result.items():
                print(f"{k.upper().ljust(15)}: {sum(v) / len(v):.3f}%")
            print("=== Tensor Active (Throughput %) ===")
            for k, v in tensor_active_result.items():
                print(f"{k.upper().ljust(15)}: {sum(v) / len(v):.3f}%")
        # print("=== Avg. Computation Time ===")
        # for k, v in avg_computation_time.items():
        #     print(f"{k.upper().ljust(15)}: {(sum(v) / len(v)) / 1000:.3f} μs, (GPU Utils: {self.calc_gpu_utils(HW_OPS=_HW_OPS, SW_OPS=_SW_OPS[k], time=sum(v) / len(v))} %)")
