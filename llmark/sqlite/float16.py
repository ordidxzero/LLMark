import sqlite3
from typing import List, Dict, Optional
from collections import OrderedDict, defaultdict
from llmark.sqlite import StateMachine, Kernel

FP16_DECODE_KERNEL_V2 = {
    1: OrderedDict({
        'cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_64x6_tn_align8>(T1::Params)': ['qkv_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_splitkv_kernel': ['self_attn'],
        'void flash_fwd_splitkv_combine_kernel': ['self_attn_end'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn': ['o_proj', 'down_proj'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['gate_up_proj', 'lm_head']
    }),
    2: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['qkv_proj', 'gate_up_proj', 'lm_head'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_splitkv_kernel': ['self_attn'],
        'void flash_fwd_splitkv_combine_kernel': ['self_attn_end'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn': ['o_proj', 'down_proj'],
    }),
    4: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['qkv_proj', 'gate_up_proj', 'lm_head'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_splitkv_kernel': ['self_attn'],
        'void flash_fwd_splitkv_combine_kernel': ['self_attn_end'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn': ['o_proj', 'down_proj'],
    }),
    8: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['qkv_proj', 'gate_up_proj', 'lm_head'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_splitkv_kernel': ['self_attn'],
        'void flash_fwd_splitkv_combine_kernel': ['self_attn_end'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn': ['o_proj', 'down_proj'],
    }),
    16: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['qkv_proj', 'gate_up_proj', 'lm_head'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_splitkv_kernel': ['self_attn'],
        'void flash_fwd_splitkv_combine_kernel': ['self_attn_end'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn': ['o_proj', 'down_proj'],
    }),
    32: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_tn': ['qkv_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, (int)4, (bool)0, (bool)0, cutlass::half_t, Flash_kernel_traits<(int)128, (int)64, (int)128, (int)4, cutlass::half_t>>, (bool)0, (bool)0, (bool)0, (bool)0, (bool)1, (bool)0, (bool)0, (bool)0>(Flash_fwd_params)': ['self_attn_end'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['o_proj', 'lm_head'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x3_tn_align8>(T1::Params)': ['gate_up_proj'],
        'ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x3_tn': ['down_proj'],
    }),
    64: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['qkv_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_splitkv_kernel': ['self_attn_end'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn': ['o_proj'],
        'ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x3_tn': ['gate_up_proj', 'down_proj', 'lm_head'],
    }),
    128: OrderedDict({
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_64x6_tn_align8>(T1::Params)': ['qkv_proj', 'o_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_splitkv_kernel': ['self_attn_end'],
        'ampere_fp16_s16816gemm_fp16_64x128_ldg8_f2f_stages_32x6_tn': ['gate_up_proj'],
        'ampere_fp16_s16816gemm_fp16_64x128_ldg8_f2f_stages_64x4_tn': ['down_proj'],
        'ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn': ['lm_head']
    }),
    256: OrderedDict({
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)': ['qkv_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64, (int)128, (int)4, (bool)0, (bool)0, cutlass::half_t, Flash_kernel_traits<(int)128, (int)64, (int)128, (int)4, cutlass::half_t>>, (bool)0, (bool)0, (bool)0, (bool)0, (bool)1, (bool)0, (bool)0, (bool)0>(Flash_fwd_params)': ['self_attn_end'],
        'ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn': ['o_proj'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8>(T1::Params)': ['gate_up_proj'],
        'ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn': ['down_proj', 'lm_head']
    }),
}

FP16_PREFILL_KERNEL_V2 = {
    1: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_128x256_ldg8_f2f_stages_64x3_tn': ['qkv_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_kernel': ['self_attn_end'],
        'ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn': ['o_proj'],
        'ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_tn': ['gate_up_proj'],
        'ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn': ['down_proj'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['lm_head']
    }),
    2: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn': ['qkv_proj', 'down_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_kernel': ['self_attn_end'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8>(T1::Params)': ['o_proj'],
        'ampere_fp16_s16816gemm_fp16_128x256_ldg8_f2f_stages_64x3_tn': ['gate_up_proj'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['lm_head']
    }),
    4: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn': ['qkv_proj', 'gate_up_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_kernel': ['self_attn_end'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8>(T1::Params)': ['o_proj'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x6_tn_align8>(T1::Params)': ['down_proj'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['lm_head']
    }),
    8: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_128x256_ldg8_f2f_stages_64x3_tn': ['qkv_proj', 'gate_up_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_kernel': ['self_attn_end'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8>(T1::Params)': ['o_proj'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x6_tn_align8>(T1::Params)': ['down_proj'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['lm_head']
    }),
    16: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn': ['qkv_proj', 'down_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_kernel': ['self_attn_end'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8>(T1::Params)': ['o_proj'],
        'ampere_fp16_s16816gemm_fp16_128x256_ldg8_f2f_stages_64x3_tn': ['gate_up_proj'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['lm_head']
    }),
    32: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn': ['qkv_proj', 'down_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_kernel': ['self_attn_end'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8>(T1::Params)': ['o_proj'],
        'ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn': ['gate_up_proj'],
        'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn': ['lm_head']
    }),
    64: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn': ['qkv_proj', 'gate_up_proj', 'down_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_kernel': ['self_attn_end'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8>(T1::Params)': ['o_proj'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)': ['lm_head']
    }),
    128: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn': ['qkv_proj', 'gate_up_proj', 'down_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_kernel': ['self_attn_end'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8>(T1::Params)': ['o_proj'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)': ['lm_head']
    }),
    256: OrderedDict({
        'ampere_fp16_s16816gemm_fp16_128x256_ldg8_f2f_stages_64x3_tn': ['qkv_proj'],
        'void vllm::reshape_and_cache_flash_kernel': ['self_attn'],
        'void flash_fwd_kernel': ['self_attn_end'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8>(T1::Params)': ['o_proj'],
        'ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn': ['gate_up_proj', 'lm_head'],
        'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x6_tn_align8>(T1::Params)': ['down_proj']
    }),
}

class Float16StateMachine(StateMachine):
    qkv_proj: int
    self_attn: List[int]
    o_proj: int
    gate_up_proj: int
    down_proj: int
    lm_head: int
    predefined_kernel: OrderedDict[str, List[str]]
    cursor: sqlite3.Cursor

    layer: int
    mapping_table: Dict[int, List[str]]
    tick_state: Dict[int, int]
    skip_state: Dict[int, bool]
    metrics: Dict[str, List[float]]
    self_attn_buffer: Optional[Kernel]
    self_attn_edge_case: List[Kernel]
    def __init__(self, cursor: sqlite3.Cursor, predefined_kernel: OrderedDict[str, List[str]]):
        self.predefined_kernel = predefined_kernel
        self.cursor = cursor
        self.self_attn = []
        self.layer = 0
        self.mapping_table = defaultdict(list)
        self.tick_state = defaultdict(int)
        self.skip_state = defaultdict(bool)
        self.metrics = defaultdict(list)
        self.self_attn_buffer = None
        self.self_attn_edge_case = []

        for kernel_name, nvtx_name in self.predefined_kernel.items():
            QUERY = f"SELECT id FROM StringIds WHERE value LIKE '%{kernel_name}%'"
            self.cursor.execute(QUERY)
            kernel_id = self.cursor.fetchone()
            assert kernel_id is not None, f"Mapping Initialization Error"
            kernel_id = kernel_id[0]
            for name in nvtx_name:
                self.mapping_table[kernel_id].append(name)
                if 'self_attn' in name:
                    self.self_attn.append(kernel_id)
                else:
                    setattr(self, name, kernel_id)

    def transform(self, kernel: Kernel):
        names = self.mapping_table.get(kernel.id, None)

        if names is None:
            return None
        
        if 'lm_head' in names:
            if len(names) == 1:
                return 'lm_head'
            
            if len(names) == 2:
                if self.tick_state[kernel.id] != 0 and self.tick_state[kernel.id] % 32 == 0 and not self.skip_state[kernel.id]:
                    self.skip_state[kernel.id] = True
                    return 'lm_head'
                
                self.skip_state[kernel.id] = False
                self.tick_state[kernel.id] += 1
                return names[0]
            if len(names) == 3:
                if self.tick_state[kernel.id] != 0 and self.tick_state[kernel.id] % 64 == 0 and not self.skip_state[kernel.id]:
                    self.skip_state[kernel.id] = True
                    return 'lm_head'

                self.skip_state[kernel.id] = False
                idx = self.tick_state[kernel.id] % (len(names) - 1)
                self.tick_state[kernel.id] += 1
                return names[idx]
        else:
            if len(names) == 1:
                return names[0]
            
            idx = self.tick_state[kernel.id] % len(names)
            self.tick_state[kernel.id] += 1
            return names[idx]
        
        return None
    
    def log(self, alias: str, kernel: Kernel):
        if alias == 'self_attn':
            self.self_attn_edge_case.append(kernel)
        
        if 'self_attn' not in alias:
            if len(self.self_attn_edge_case) >= 2:
                self.metrics['self_attn'].append(self.self_attn_edge_case[-1].end - self.self_attn_edge_case[0].start)
                self.self_attn_edge_case = []
            self.metrics[alias].append(kernel.end - kernel.start)
        else:
            if self.self_attn_buffer is None and alias == 'self_attn':
                self.self_attn_buffer = kernel

            if self.self_attn_buffer is not None and alias == 'self_attn_end':
                self.metrics['self_attn'].append(kernel.end - self.self_attn_buffer.start)
                self.self_attn_buffer = None
                self.self_attn_edge_case = []

    def get_metrics(self):
        return self.metrics
    
    def reset(self):
        self.self_attn = []
        self.layer = 0
        self.tick_state = defaultdict(int)
        self.skip_state = defaultdict(bool)
        self.metrics = defaultdict(list)
        self.self_attn_buffer = None
        self.self_attn_edge_case = []