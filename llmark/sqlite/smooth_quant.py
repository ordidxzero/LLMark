from typing import Literal
from llmark.sqlite import StateMachine, Kernel

class SmoothQuantStateMachine(StateMachine):
    def __init__(self, phase: Literal['prefill', 'decode'] = 'decode'):
        self._quant_piece = 'void vllm::dynamic_scaled_int8_quant_kernel'
        self._proj_piece = 'void cutlass::Kernel2<vllm::enable_sm80_to_sm89<cutlass::gemm::kernel::DefaultGemmWithVisitor'
        self._lm_head_piece = ['ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn', 'ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x3_tn', 'ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn']
        self._cache_piece = 'void vllm::reshape_and_cache_flash_kernel'
        self._flash_attn_piece = 'void flash_fwd_splitkv_kernel'
        self._flash_combine_piece = 'void flash_fwd_splitkv_combine_kernel'

        if phase == 'prefill':
            self._flash_combine_piece = 'void flash_fwd_kernel'
            self._flash_attn_piece = '⋅⋅⋅⋅'

        self.cycle = 0
        self._metrics = {
            "self_attn": []
        }
        self.self_attn_state = {
            "start_kernel": None,
            "edge_case": []
        }

    def transform(self, kernel: Kernel):
        if self._cache_piece in kernel.name or self._flash_attn_piece in kernel.name:
            return 'self_attn'
        
        if self._flash_combine_piece in kernel.name:
            return 'self_attn_end'
        
        cycle = self.cycle % 4
        if cycle == 0:
            if self._quant_piece in kernel.name:
                return 'qkv_quant'
            if self._proj_piece in kernel.name:
                self.cycle += 1
                return 'qkv_proj'
        
        if cycle == 1:
            if self._quant_piece in kernel.name:
                return 'o_quant'
            if self._proj_piece in kernel.name:
                self.cycle += 1
                return 'o_proj'
            
        if cycle == 2:
            if self._quant_piece in kernel.name:
                return 'gate_up_quant'
            if self._proj_piece in kernel.name:
                self.cycle += 1
                return 'gate_up_proj'
        
        if cycle == 3:
            if self._quant_piece in kernel.name:
                return 'down_quant'
            if self._proj_piece in kernel.name:
                self.cycle = 0
                return 'down_proj'
        
        if kernel.name in self._lm_head_piece:
            return 'lm_head'
        
        return None
    
    def log(self, alias: str, kernel: Kernel):
        if alias == 'self_attn':
            self.self_attn_state['edge_case'].append(kernel)

        if 'self_attn' not in alias:
            if len(self.self_attn_state['edge_case']) >= 2:
                self._metrics['self_attn'].append(self.self_attn_state['edge_case'][-1].end - self.self_attn_state['edge_case'][0].start)
                self.self_attn_state['edge_case'] = []
            if alias not in self._metrics:
                self._metrics[alias] = [kernel.end - kernel.start]
            else:
                self._metrics[alias].append(kernel.end - kernel.start)
        else:
            if self.self_attn_state['start_kernel'] is None and alias == 'self_attn':
                self.self_attn_state['start_kernel'] = kernel
            
            if self.self_attn_state['start_kernel'] is not None and alias == 'self_attn_end':
                self._metrics['self_attn'].append(kernel.end - self.self_attn_state['start_kernel'].start)
                self.self_attn_state['start_kernel'] = None
                self.self_attn_state["edge_case"] = []

    def get_metrics(self):
        return self._metrics
    
    def reset(self):
        self.cycle = 0
        self._metrics = {
            "self_attn": []
        }
        self.self_attn_state = {
            "start_kernel": None,
            "edge_case": []
        }