from typing import List, Optional, Literal
from collections import defaultdict
from llmark.sqlite import StateMachine, Kernel

class GPTQMarlinStateMachine(StateMachine):
    marlin_kernel_buffer: Optional[Kernel]
    self_attn_buffer: Optional[Kernel]
    self_attn_edge_case: List[Kernel]
    def __init__(self, phase: Literal['prefill', 'decode'] = 'decode'):
        self.marlin_kernel = 'void marlin::Marlin'
        self.cache_piece = 'void vllm::reshape_and_cache_flash_kernel'
        self.flash_attn_piece = 'void flash_fwd_splitkv_kernel'

        if phase == 'prefill':
            self.flash_attn_piece = 'void flash_fwd_kernel'

        self.k_alias = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj']

        self.metrics = defaultdict(list)
        self.self_attn_buffer = None
        self.self_attn_edge_case = []

        self.cycle = 0
        self.current_alias = None
        self.marlin_kernel_buffer = None
        self.prev_marlin_kernel = None

    def transform(self, kernel: Kernel):
        if self.cache_piece in kernel.name:
            return 'self_attn'
        
        if self.flash_attn_piece in kernel.name:
            return 'self_attn_end'
        
        if self.marlin_kernel in kernel.name:
            self.prev_marlin_kernel = kernel
            if self.current_alias is None:
                self.current_alias = self.k_alias[self.cycle % 4]
                self.marlin_kernel_buffer = kernel
            return None
        
        if self.current_alias is None:
            return None

        if self.current_alias is not None and self.marlin_kernel_buffer is not None and self.prev_marlin_kernel is not None:
            # Marlin 커널이 끝난 후 연산.
            new_kernel = Kernel(self.marlin_kernel_buffer.start, self.prev_marlin_kernel.end, self.prev_marlin_kernel.id, self.prev_marlin_kernel.name)
            alias = self.current_alias
            self.current_alias = None
            self.prev_marlin_kernel = None
            self.marlin_kernel_buffer = None
            self.cycle += 1
            return alias, new_kernel
            
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
        self.current_alias = None
        self.prev_marlin_kernel = None
        self.marlin_kernel_buffer = None
        self.cycle = 0
        self.metrics = defaultdict(list)
        self.self_attn_buffer = None
        self.self_attn_edge_case = []