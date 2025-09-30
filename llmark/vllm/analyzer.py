import re, json
from pathlib import Path
from llmark.utils import LogAnalyzer
from typing import List, Tuple, Optional
from tqdm import tqdm

def plot_sequential_data(blocks: List[int | float], filename: str, target_path: Path, title: str = 'GPU KV Cache Usage', ylabel: str = 'Percentage', ylim: int = 100):
    import matplotlib.pyplot as plt
    x = list(range(len(blocks)))  # x축 인덱스

    plt.figure(figsize=(10, 5))
    plt.bar(x, blocks, width=1.0, align='edge')
    plt.ylabel(ylabel)
    plt.ylim(0, ylim)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    p = target_path / filename
    # 저장: 파일명과 확장자를 지정 (예: PNG, JPG, PDF 등)
    plt.savefig(p.absolute().as_posix(), dpi=300)  # 고해상도로 저장
    plt.close()  # 메모리 절약을 위해 닫기

class BaseState:
    def mean(self, d: List[int | float]):
        return round(sum(d) / len(d), 2)

class VLLMPromptState(BaseState):
    iter_nums: List[int]
    running_seqs: List[int]
    latencies: List[float]
    latencies_with_delay: List[float]
    latencies_without_delay: List[float]
    prompt_delay_list: List[float]
    prompt_delay: float
    is_prompt_group: bool
    prompt_buffer: Optional[Tuple[int, int, float]]
    prompt_group: List[Tuple[int, int, float]]
    prompt_group_list: List[List[Tuple[int, int, float]]]

    def get_avg_running_seqs(self):
        return self.mean(self.running_seqs)
    
    def get_avg_latency(self):
        return self.mean(self.latencies)
    
    def get_total_latency(self):
        return round(sum(self.latencies), 2)
    
    def get_avg_latency_with_delay(self):
        return self.mean(self.latencies_with_delay)
    
    def get_avg_latency_without_delay(self):
        return self.mean(self.latencies_without_delay)
    
    def get_avg_prompt_delay(self):
        return self.mean(self.prompt_delay_list)
    
    def get_total_prompt_delay(self):
        return round(sum(self.prompt_delay_list), 2)
    
    def get_avg_prompt_delay(self):
        return self.mean(self.prompt_delay_list)
    
    def __init__(self):
        self.iter_nums = []
        self.running_seqs = []
        self.latencies = []
        self.latencies_with_delay = []
        self.latencies_without_delay = []
        self.prompt_delay_list = []
        self.prompt_delay = 0
        self.is_prompt_group = False
        self.prompt_group = []
        self.prompt_group_list = []

    def push(self, iter_num: int, running_seqs: int, latency: float):
        self.iter_nums.append(iter_num)
        self.running_seqs.append(running_seqs)
        self.latencies.append(latency)


        if self.is_prompt_group:
            prompt_buffer = (iter_num, running_seqs, latency + self.prompt_delay)
            self.prompt_group.append(prompt_buffer)
        self.push_prompt_delay(latency)

    def push_prompt_delay(self, latency: float):
        if self.prompt_delay == 0: return

        self.latencies_without_delay.append(latency)
        self.latencies_with_delay.append(latency + self.prompt_delay)
        self.prompt_delay_list.append(self.prompt_delay)
        self.prompt_delay = 0

    def update_prompt_delay(self, delay: float):
        self.prompt_delay += delay

    def start_prompt_group(self):
        self.is_prompt_group = True

    def end_prompt_group(self):
        if self.is_prompt_group:
            if len(self.prompt_group) > 1:
                self.prompt_group_list.append(self.prompt_group)

            self.prompt_group = []
            self.is_prompt_group = False

    def get_prompt_group_stats(self):
        if len(self.prompt_group_list) == 0:
            return {
            "avg_running_seqs": 0,
            "avg_group_latency": 0,
            "num_groups": 0
            }
        
        running_seqs = []
        latencies = []

        for prompt_group in self.prompt_group_list:
            _, running_seqs_in_group, latencies_in_group = [list(l) for l in zip(*prompt_group)]
            running_seqs.append(sum(running_seqs_in_group))
            latencies.append(sum(latencies_in_group))
        
        return {
            "avg_running_seqs": self.mean(running_seqs),
            "avg_group_latency": self.mean(latencies),
            "num_groups": len(running_seqs)
        }
    
    def get_prompt_frequency(self):
        return self.mean(self._get_prompt_iter_gaps())

    def _get_prompt_iter_gaps(self):
        gaps = []
        for i in range(1, len(self.iter_nums)):
            prev_iter = self.iter_nums[i - 1]
            cur_iter = self.iter_nums[i]
            gaps.append(cur_iter - prev_iter)

        return gaps

class VLLMGenerationState(BaseState):
    iter_nums: List[int]
    running_seqs: List[int]
    latencies: List[float]

    def __init__(self):
        self.iter_nums = []
        self.running_seqs = []
        self.latencies = []

    def get_avg_running_seqs(self):
        return self.mean(self.running_seqs)
    
    def get_avg_latency(self):
        return self.mean(self.latencies)

    def push(self, iter_num: int, running_seqs: int, latency: float):
        self.iter_nums.append(iter_num)
        self.running_seqs.append(running_seqs)
        self.latencies.append(latency)

class VLLMMiscellaneousState(BaseState):
    original_running_seqs: List[int]
    running_seqs: List[int]
    preemption_seqs: List[int]
    gpu_total_block: Optional[int]
    gpu_kv_cache_usage: List[float]

    def __init__(self):
        self.original_running_seqs = []
        self.running_seqs = []
        self.preemption_seqs = []
        self.gpu_total_block = None
        self.gpu_kv_cache_usage = []

    def get_avg_original_running_seqs(self):
        return self.mean(self.running_seqs)
    
    def get_avg_running_seqs(self):
        return self.mean(self.running_seqs)
    
    def get_avg_preemption_seqs(self):
        return self.mean(self.preemption_seqs)

    def push(self, original_running_seqs: int, preemption_seqs: int, gpu_kv_cache_usage: float):
        self.original_running_seqs.append(original_running_seqs)
        self.preemption_seqs.append(preemption_seqs)
        self.gpu_kv_cache_usage.append(gpu_kv_cache_usage)

    def push_running_seqs(self, running_seqs: int):
        self.running_seqs.append(running_seqs)

    def update_gpu_block(self, gpu_block: int):
        if self.gpu_total_block == None:
            self.gpu_total_block = gpu_block

        return
    
    def plot_gpu_kv_cache_usage(self, name: str, save_dir: Path):
        plot_sequential_data(self.gpu_kv_cache_usage, name, save_dir)

    def plot_running_seqs(self, name: str, save_dir: Path, title="Running Seqs", ylabel='Count', ylim=256):
        plot_sequential_data(self.running_seqs, name, save_dir, title, ylabel, ylim)
    

class VLLMAnalyzerState(BaseState):
    prompt_state: VLLMPromptState
    generation_state: VLLMGenerationState
    miscellaneous_state: VLLMMiscellaneousState

    def __init__(self):
        self.prompt_state = VLLMPromptState()
        self.generation_state = VLLMGenerationState()
        self.miscellaneous_state = VLLMMiscellaneousState()

    def get_avg_running_seqs(self):
        return self.mean(self.prompt_state.running_seqs + self.generation_state.running_seqs)
    
    def get_avg_original_running_seqs(self):
        return self.miscellaneous_state.get_avg_original_running_seqs()
    
    def get_avg_prompt_seqs(self):
        return self.prompt_state.get_avg_running_seqs()
    
    def get_avg_generation_seqs(self):
        return self.generation_state.get_avg_running_seqs()
    
    def get_avg_prompt_latency(self):
        return self.prompt_state.get_avg_latency()
    
    def get_avg_generation_latency(self):
        return self.generation_state.get_avg_latency()
    
    def get_preemption_seqs(self):
        return sum(self.miscellaneous_state.preemption_seqs)
    

class VLLMAnalyzer(LogAnalyzer):
    def __init__(self, log_dir: Path):
        super().__init__(log_dir)

    def _parse_iter_line(self, line: str):
        result = re.search(r'iter: (\d+), Avg prompt throughput: (\d+.\d+) tokens/s, Avg generation throughput: (\d+.\d+) tokens/s, Running: (\d+) reqs, Swapped: (\d+) reqs, Pending: (\d+) reqs, GPU KV cache usage: (\d+.\d+)%, CPU KV cache usage: (\d+.\d+)%, GPU Total Block: (\d+), GPU Free Block: (\d+), CPU Total Block: (\d+), CPU Free Block: (\d+), actual_num_batched_tokens: (\d+), Prompt stage: (\d+) reqs, Generation stage: (\d+) reqs, Preemption: (\d+) reqs, Generation: (\d+) tokens, Latency: (\d+.\d+)', line)
        assert result is not None, "ParseError"

        iter_num = int(result.group(1))
        prompt_throughput = float(result.group(2))
        generation_throughput = float(result.group(3))
        running_seqs = int(result.group(4))
        # swapped_seqs = int(result.group(5))
        # pending_seqs = int(result.group(6))
        gpu_kv_cache_usage = float(result.group(7))
        # cpu_kv_cache_usage = float(result.group(8))
        gpu_total_block = int(result.group(9))
        # gpu_free_block = int(result.group(10))
        # cpu_total_block = int(result.group(11))
        # cpu_free_block = int(result.group(12))
        # actual_num_batched_tokens = int(result.group(13))
        prompt_seqs = int(result.group(14))
        generation_seqs = int(result.group(15))
        preemption_seqs = int(result.group(16))
        # generation_tokens = int(result.group(17))
        latency_iter = float(result.group(18))

        return iter_num, prompt_throughput, generation_throughput, gpu_kv_cache_usage, gpu_total_block, running_seqs, prompt_seqs, generation_seqs, preemption_seqs, latency_iter

    def run(self, pattern: str, save_dir: Path):
        assert pattern is not None and pattern != "" and isinstance(pattern, str), "Error"

        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        files = list(self._log_dir.glob(pattern))
        
        for logfile_path in tqdm(files):
            state = VLLMAnalyzerState()
            
            with open(logfile_path, "r") as logfile:
                for line in logfile:

                    if "metrics.py" in line and 'iter:' in line:
                        iter_num, prompt_throughput, generation_throughput, gpu_kv_cache_usage, gpu_total_block, running_seqs, prompt_seqs, generation_seqs, preemption_seqs, latency = self._parse_iter_line(line)

                        if prompt_throughput == 0.0 and generation_throughput == 0.0:
                            state.prompt_state.update_prompt_delay(latency)
                            continue

                        state.miscellaneous_state.push(running_seqs, preemption_seqs, gpu_kv_cache_usage)
                        state.miscellaneous_state.update_gpu_block(gpu_total_block)
                        
                        if prompt_throughput != 0.0:
                            # Prompt Stage
                            state.prompt_state.start_prompt_group()

                            state.prompt_state.push(iter_num, prompt_seqs, latency)
                            state.miscellaneous_state.push_running_seqs(prompt_seqs)

                        elif generation_throughput != 0.0:
                            # Generation Stage
                            state.generation_state.push(iter_num, generation_seqs, latency)
                            state.miscellaneous_state.push_running_seqs(generation_seqs)

                            state.prompt_state.end_prompt_group()

                        else:
                            continue

            def get_plot_name(logfile_path: Path, suffix: str):
                return logfile_path.with_stem(logfile_path.stem + "_" + suffix).with_suffix(".png").name

            # state.miscellaneous_state.plot_gpu_kv_cache_usage(get_plot_name(logfile_path, "kvcache_usage"), save_dir)
            # state.miscellaneous_state.plot_running_seqs(get_plot_name(logfile_path, "running_seqs"), save_dir, title="Running Seqs", ylabel='Count', ylim=256)

            with open(save_dir / f"{logfile_path.stem}.json", "w") as output_file:
                content = {
                    "original_running_seqs": state.get_avg_original_running_seqs(),
                    "avg_running_seqs": state.get_avg_running_seqs(),
                    "avg_prompt_seqs": state.get_avg_prompt_seqs(),
                    "avg_generation_seqs": state.get_avg_generation_seqs(),
                    "avg_prompt_latency": state.get_avg_prompt_latency(),
                    "avg_generation_latency": state.get_avg_generation_latency(),
                    "avg_prompt_delay": state.prompt_state.get_avg_prompt_delay(),
                    "avg_prompt_latency_with_delay": state.prompt_state.get_avg_latency_with_delay(),
                    "avg_prompt_latency_without_delay": state.prompt_state.get_avg_latency_without_delay(),
                    "preemption_seqs": state.get_preemption_seqs(),
                    "gpu_total_block": state.miscellaneous_state.gpu_total_block,
                    "prefill_frequency": state.prompt_state.get_prompt_frequency(),
                    "total_prompt_latency": state.prompt_state.get_total_latency(),
                    "total_prompt_delay": state.prompt_state.get_total_prompt_delay(),
                    "num_stalls_by_prompt": len(state.prompt_state.iter_nums),
                    "prompt_groups": state.prompt_state.get_prompt_group_stats()
                }

                json.dump(content, output_file, indent=2)