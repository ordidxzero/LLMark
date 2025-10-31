import argparse, re, sqlite3
from typing import List, Tuple, Dict
from tqdm import tqdm
from collections import defaultdict, OrderedDict


class SmoothQuantStateMachine:
    def __init__(self):
        self._quant_piece = 'void vllm::dynamic_scaled_int8_quant_kernel'
        self._proj_piece = 'void cutlass::Kernel2<vllm::enable_sm80_to_sm89<cutlass::gemm::kernel::DefaultGemmWithVisitor'
        self._lm_head_piece = 'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn'
        self._cache_piece = 'reshape_and_cache_flash_kernel'
        self._flash_attn_piece = 'flash_fwd_splitkv_kernel'
        self._flash_combine_piece = 'flash_fwd_splitkv_combine_kernel'
        self.cycle = 0

    def transform(self, kernel_name: str):
        if self._cache_piece in kernel_name or self._flash_attn_piece in kernel_name:
            return 'self_attn'
        
        if self._flash_combine_piece in kernel_name:
            return 'self_attn_end'
        
        cycle = self.cycle % 4
        if cycle == 0:
            if self._quant_piece in kernel_name:
                return 'qkv_quant'
            if self._proj_piece in kernel_name:
                self.cycle += 1
                return 'qkv_proj'
        
        if cycle == 1:
            if self._quant_piece in kernel_name:
                return 'o_quant'
            if self._proj_piece in kernel_name:
                self.cycle += 1
                return 'o_proj'
            
        if cycle == 2:
            if self._quant_piece in kernel_name:
                return 'gate_up_quant'
            if self._proj_piece in kernel_name:
                self.cycle += 1
                return 'gate_up_proj'
        
        if cycle == 3:
            if self._quant_piece in kernel_name:
                return 'down_quant'
            if self._proj_piece in kernel_name:
                self.cycle += 1
                return 'down_proj'
        
        if self._lm_head_piece in kernel_name:
            return 'lm_head'
        
        return None
    
    def reset(self):
        self.cycle = 0


FP16_DECODE_KERNEL = {
    1: OrderedDict({
        'cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_64x6_tn_align8>(T1::Params)': ['qkv_proj'],
        'reshape_and_cache_flash_kernel': ['self_attn'],
        'flash_fwd_splitkv_kernel': ['self_attn'],
        'flash_fwd_splitkv_combine_kernel': ['self_attn_end'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn': ['o_proj', 'down_proj'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['gate_up_proj', 'lm_head']
    })
}

class Float16StateMachine:
    _i: int
    _state: Dict[int, List[str]]
    _tick: Dict[int, int]
    _skip: Dict[int, bool]
    def __init__(self, max_num_seqs: int):
        self._decode_kernel = FP16_DECODE_KERNEL.get(max_num_seqs)
        assert self._decode_kernel is not None, "Error"
        self._i = 0
        self._state = defaultdict(list)
        self._tick = defaultdict(int)
        self._skip = defaultdict(bool)
    
    def set_kernel_id(self, kernel_id:int, name: str):
        self._state[kernel_id].append(name)

    def get_name(self, kernel_id: int):
        names = self._state.get(kernel_id, None)
        if names is None:
            return None
        if 'lm_head' in names:
            if len(names) == 1:
                return 'lm_head'
            
            if len(names) == 2:
                if self._tick[kernel_id] != 0 and self._tick[kernel_id] % 32 == 0 and not self._skip[kernel_id]:
                    self._skip[kernel_id] = True
                    return 'lm_head'
                
                self._skip[kernel_id] = False
                self._tick[kernel_id] += 1
                return names[0]
            if len(names) == 3:
                if self._tick[kernel_id] != 0 and self._tick[kernel_id] % 64 == 0 and not self._skip[kernel_id]:
                    self._skip[kernel_id] = True
                    return 'lm_head'

                self._skip[kernel_id] = False
                idx = self._tick[kernel_id] % (len(names) - 1)
                self._tick[kernel_id] += 1
                return names[idx]
        else:
            if len(names) == 1:
                return names[0]
            
            idx = self._tick[kernel_id] % len(names)
            self._tick[kernel_id] += 1
            return names[idx]
        
        return None

class KernelLogger:
    _cursor: sqlite3.Cursor
    _prefill_kernels: List[str]
    _last_prefill_end_time: int
    _decode_start_kernel: str
    def __init__(self, cursor: sqlite3.Cursor, max_num_seqs: int):
        self._cursor = cursor

        QUERY = "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE '%PREFILL%' AND text NOT LIKE '%reqs: 0)%' AND text NOT LIKE '%reqs: 1)%' ORDER BY start ASC"
        if max_num_seqs == 1:
            QUERY = "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE '%PREFILL%' AND text NOT LIKE '%reqs: 0)%' ORDER BY start ASC LIMIT 100 OFFSET 1"
        self._cursor.execute(QUERY)

        prefill_nvtx_events = self._cursor.fetchall()

        last_prefill = prefill_nvtx_events[-1]
        prefill_start_time = last_prefill[0]
        prefill_end_time = last_prefill[1]

        QUERY = f"SELECT kernel.start, kernel.end, s.value FROM StringIds AS s JOIN KERNEL_WITH_INDEX AS kernel ON s.id = kernel.demangledName WHERE kernel.correlationId IN (SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >= {prefill_start_time} AND end <= {prefill_end_time + 6000000})"
        self._cursor.execute(QUERY)
        self._prefill_kernels = self._cursor.fetchall()
        self._decode_start_kernel = 'indexSelectLargeIndex' if max_num_seqs >= 32 else 'indexSelectSmallIndex'

        self._last_prefill_end_time = prefill_end_time
    
    def _fprint(self, i: int, kernel: Tuple[int, int, str]):
        start_time, end_time, kernel_name = kernel
        print(f"{i}".ljust(4), ":", start_time, end_time, kernel_name)

    def _iter_print(self, title: str, kernels: List[Tuple[int, int, str]]):
        print("=" * 10, title, "=" * 10)
        for i, kernel in enumerate(kernels[:30]):
            self._fprint(i, kernel)
        print('--- skip ---')
        for i, kernel in enumerate(kernels[-30:]):
            self._fprint(i, kernel)

    def print_prefill_kernel(self):
        self._iter_print('Prefill', self._prefill_kernels) # type: ignore

    def print_decode_kernel(self):
        QUERY = f"SELECT kernel.start, kernel.end, s.value FROM StringIds AS s JOIN KERNEL_WITH_INDEX AS kernel ON s.id = kernel.demangledName WHERE kernel.correlationId IN (SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >= {self._last_prefill_end_time}) ORDER BY start ASC"
        self._cursor.execute(QUERY)
        result = self._cursor.fetchall()
        kernels = []
        is_invalid_kernel = True
        for k in result:
            start_time, end_time, kernel_name = k
            kernels.append((start_time, end_time, kernel_name))
            if 'void at::native::reduce_kernel' in kernel_name:
                if is_invalid_kernel:
                    is_invalid_kernel = False
                else:
                    break

        self._iter_print('Decode', kernels)
    
    def calc_decode_kernel(self):
        is_first = True
        durations = []
        pbar = tqdm(total=2048)
        idx = 0
        QUERY = f"SELECT kernel.start, kernel.end, s.value FROM StringIds AS s JOIN KERNEL_WITH_INDEX AS kernel ON s.id = kernel.demangledName WHERE kernel.correlationId IN (SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >= {self._last_prefill_end_time}) ORDER BY start ASC"
        self._cursor.execute(QUERY)
        result = self._cursor.fetchall()
        state_machine = SmoothQuantStateMachine()
        metrics = defaultdict(list)
        while True:
            kernels = []
            if is_first:
                is_valid_kernels = None
                for k in result:
                    idx += 1
                    start_time, end_time, kernel_name = k
                    if self._decode_start_kernel in kernel_name and is_valid_kernels == False:
                        is_valid_kernels = True
                    if is_valid_kernels:
                        kernels.append((start_time, end_time, kernel_name))
                    if 'void at::native::reduce_kernel' in kernel_name and is_valid_kernels == None:
                        is_valid_kernels = False
                    if 'void at::native::reduce_kernel' in kernel_name and is_valid_kernels == True:
                        break
                    
                is_first = False
            else:
                if len(result[idx:]) == 0:
                    break
                is_start = False
                for k in result[idx:]:
                    idx += 1
                    start_time, end_time, kernel_name = k
                    if not is_start and self._decode_start_kernel in kernel_name:
                        is_start = True

                    if is_start:
                        kernels.append((start_time, end_time, kernel_name))

                    if is_start and 'void at::native::reduce_kernel' in kernel_name:
                        break

            
            _metrics = {
                "self_attn": []
            }
            self_attn_temp = 0
            _self_attn_edge_case = []
            for kernel in kernels:
                start_time, end_time, kernel_name = kernel
                name = state_machine.transform(kernel_name)

                if name is None:
                    continue

                if name == 'self_attn':
                    _self_attn_edge_case.append((start_time, end_time))

                if 'self_attn' not in name:
                    if len(_self_attn_edge_case) == 2:
                        _metrics['self_attn'].append(_self_attn_edge_case[1][1] - _self_attn_edge_case[0][0])
                        _self_attn_edge_case = []
                    if name not in _metrics:
                        _metrics[name] = [end_time - start_time]
                    else:
                        _metrics[name].append(end_time - start_time)
                else:
                    if self_attn_temp == 0 and name == 'self_attn':
                        self_attn_temp = start_time
                    
                    if self_attn_temp != 0 and name == 'self_attn_end':
                        _metrics['self_attn'].append(end_time - self_attn_temp)
                        self_attn_temp = 0

            for k, v in _metrics.items():
                if k == 'lm_head':
                    metrics[k].append(v)
                else:
                    metrics[k].append(sum(v))

            pbar.update(1)
            start_time = kernels[0][0]
            end_time = kernels[-1][1]
            durations.append(round((end_time - start_time) / 1000000, 2))
            state_machine.reset()
        print("TPOT                     : ", round(sum(durations) / len(durations), 5))
        print("QKV PROJ                 : ", round(sum(metrics['qkv_proj']) / len(metrics['qkv_proj']), 5))
        print("SELF ATTN                : ", round(sum(metrics['self_attn']) / len(metrics['self_attn']), 5))
        print("O PROJ                   : ", round(sum(metrics['o_proj']) / len(metrics['o_proj']), 5))
        print("GATE UP PROJ             : ", round(sum(metrics['gate_up_proj']) / len(metrics['gate_up_proj']), 5))
        print("DOWM PROJ                : ", round(sum(metrics['down_proj']) / len(metrics['down_proj']), 5))
        print("QUANT OVERHEAD (qkv)     : ", round(sum(metrics['qkv_quant']) / len(metrics['qkv_quant']), 5))
        print("QUANT OVERHEAD (o)       : ", round(sum(metrics['o_quant']) / len(metrics['o_quant']), 5))
        print("QUANT OVERHEAD (gate_up) : ", round(sum(metrics['gate_up_quant']) / len(metrics['gate_up_quant']), 5))
        print("QUANT OVERHEAD (down)    : ", round(sum(metrics['down_quant']) / len(metrics['down_quant']), 5))

def connect(filename: str):
    connection = sqlite3.connect(filename)
    cursor = connection.cursor()

    return cursor, connection

def init(cursor: sqlite3.Cursor, connection: sqlite3.Connection):
    cursor.execute("CREATE TABLE IF NOT EXISTS KERNEL_WITH_INDEX AS SELECT start, end, demangledName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL")
    # cursor.execute("ALTER TABLE KERNEL_WITH_INDEX ADD COLUMN start_bucket INTEGER GENERATED ALWAYS AS (FLOOR(start / 1000000)) VIRTUAL")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_kernel_start ON KERNEL_WITH_INDEX(start)")

    cursor.execute("CREATE TABLE IF NOT EXISTS VIRTUAL_NVTX_OPS (pid INTEGER PRIMARY KEY, name TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL)")
    cursor.execute("CREATE TABLE IF NOT EXISTS VIRTUAL_NVTX_EVENTS (pid INTEGER PRIMARY KEY, name TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL)")
    connection.commit()
    return

def get_max_num_seqs(filename: str):
    nums = re.search(r"max-num-seqs_(\d+)_num-prompts_(\d+)", filename)
    assert nums is not None, "error"
    max_num_seqs = int(nums.group(1))
    num_prompts = int(nums.group(2))
    return max_num_seqs, int(num_prompts / max_num_seqs)


def main(args: argparse.Namespace):
    cursor, connection = connect(args.filename)

    init(cursor, connection)

    max_num_seqs, _ = get_max_num_seqs(args.filename)
    
    logger = KernelLogger(cursor=cursor, max_num_seqs=max_num_seqs)

    if args.phase == 'decode':
        if args.calc:
            logger.calc_decode_kernel()
        else:
            logger.print_decode_kernel()
    
    if args.phase == 'prefill':
        logger.print_prefill_kernel()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("filename")
    parser.add_argument("phase", choices=['decode', 'prefill'])
    parser.add_argument("--calc", action='store_true')

    args = parser.parse_args()

    main(args)