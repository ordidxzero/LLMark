import re, json
from pathlib import Path
from llmark.utils import LogAnalyzer
from typing import List


class VLLMAnalyzer(LogAnalyzer):
    def __init__(self, log_dir: Path):
        super().__init__(log_dir)
    
    def _parse_line(self, line: str):
        result = re.search(r'iter: (\d+), Avg prompt throughput: (\d+.\d+) tokens/s, Avg generation throughput: (\d+.\d+) tokens/s, Running: (\d+) reqs, Swapped: (\d+) reqs, Pending: (\d+) reqs, GPU KV cache usage: (\d+.\d+)%, CPU KV cache usage: (\d+.\d+)%, GPU Total Block: (\d+), GPU Free Block: (\d+), CPU Total Block: (\d+), CPU Free Block: (\d+), actual_num_batched_tokens: (\d+), Prompt stage: (\d+) reqs, Generation stage: (\d+) reqs, Preemption: (\d+) reqs, Generation: (\d+) tokens, Latency: (\d+.\d+)', line)
        
        iter_num = int(result.group(1))
        prompt_throughput = float(result.group(2))
        generation_throughput = float(result.group(3))
        # running_seqs = int(result.group(4))
        # swapped_seqs = int(result.group(5))
        # pending_seqs = int(result.group(6))
        gpu_kv_cache_usage = float(result.group(7))
        # cpu_kv_cache_usage = float(result.group(8))
        gpu_total_block = int(result.group(9))
        gpu_free_block = int(result.group(10))
        cpu_total_block = int(result.group(11))
        cpu_free_block = int(result.group(12))
        actual_num_batched_tokens = int(result.group(13))
        prompt_seqs = int(result.group(14))
        generation_seqs = int(result.group(15))
        preemption_seqs = int(result.group(16))
        generation_tokens = int(result.group(17))
        latency_iter = float(result.group(18))

        return iter_num, prompt_throughput, generation_throughput, gpu_kv_cache_usage, gpu_total_block, gpu_free_block, prompt_seqs, generation_seqs, preemption_seqs, latency_iter
    
    def _get_prompt_iter_gap(self, iter_nums: List[int]):
        gaps = []
        for i in range(1, len(iter_nums)):
            prev_iter = iter_nums[i - 1]
            cur_iter = iter_nums[i]
            gaps.append(cur_iter - prev_iter)

        return gaps
    
    def _mean(self, d: List[int | float]):
        return round(sum(d) / len(d), 2)
    
    def plot_gpu_blocks(self, blocks: List[int], source_path: Path, target_path: Path):
        import matplotlib.pyplot as plt

        x = list(range(len(blocks)))  # x축 인덱스

        plt.figure(figsize=(10, 5))
        plt.bar(x, blocks, width=1.0, align='edge')

        plt.ylabel('Percentage')
        plt.ylim(0, 100)
        plt.title('GPU KV Cache Usage')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        p = target_path / source_path.with_suffix(".png").name

        # 저장: 파일명과 확장자를 지정 (예: PNG, JPG, PDF 등)
        plt.savefig(p.absolute().as_posix(), dpi=300)  # 고해상도로 저장
        plt.close()  # 메모리 절약을 위해 닫기

    def run(self, pattern: str, save_dir: Path = None):
        assert pattern is not None and pattern != "" and isinstance(pattern, str), "Error"
        
        for logfile_path in self._log_dir.glob(pattern):
            prompt_state = {
                "iter_nums": [],
                "running_seqs": [],
                "latencies": [],
            }
            generation_state = {
                "iter_nums": [],
                "running_seqs": [],
                "latencies": []
            }
            rest_state = {
                "preemption_seqs": [],
                "gpu_total_block": -1,
                "gpu_kv_cache_usage": []
            }
            with open(logfile_path, "r") as logfile:
                is_end_test_request = None
                test_end_index = 0
                for line in logfile:
                    if "metrics.py" in line:
                        iter_num, prompt_throughput, generation_throughput, gpu_kv_cache_usage, gpu_total_block, gpu_free_block, prompt_seqs, generation_seqs, preemption_seqs, latency = self._parse_line(line)
                        rest_state['preemption_seqs'].append(preemption_seqs)
                        rest_state['gpu_kv_cache_usage'].append(gpu_kv_cache_usage)

                        if is_end_test_request != True:
                            test_end_index += 1

                        if prompt_seqs == 0 and generation_seqs == 0:
                            if is_end_test_request is None:
                                is_end_test_request = False
                            elif is_end_test_request == False:
                                is_end_test_request = True

                        if rest_state['gpu_total_block'] == -1:
                            rest_state['gpu_total_block'] = gpu_total_block
                        
                        if prompt_throughput != 0.0:
                            # Prompt Stage
                            prompt_state["iter_nums"].append(iter_num)
                            prompt_state['running_seqs'].append(prompt_seqs)
                            prompt_state['latencies'].append(latency)
                        elif generation_throughput != 0.0:
                            # Generation Stage
                            generation_state["iter_nums"].append(iter_num)
                            generation_state['running_seqs'].append(generation_seqs)
                            generation_state['latencies'].append(latency)
                        else:
                            continue

            all_running_seqs = generation_state['running_seqs'] + prompt_state['running_seqs']
            prompt_iter_gaps = self._get_prompt_iter_gap(prompt_state['iter_nums'])


            dist_dir = save_dir if save_dir is not None else self._log_dir

            self.plot_gpu_blocks(rest_state['gpu_kv_cache_usage'][test_end_index:], logfile_path, save_dir)

            if not dist_dir.exists():
                dist_dir.mkdir(parents=True)

            with open(dist_dir / f"{logfile_path.stem}.json", "w") as output_file:
                content = {
                    "avg_running_seqs": self._mean(all_running_seqs),
                    "avg_prompt_seqs": self._mean(prompt_state['running_seqs']),
                    "avg_generation_seqs": self._mean(generation_state['running_seqs']),
                    "avg_prompt_latency": self._mean(prompt_state['latencies']),
                    "avg_generation_latency": self._mean(generation_state['latencies']),
                    "preemption_seqs": sum(rest_state['preemption_seqs']),
                    "gpu_total_block": rest_state['gpu_total_block'],
                    "prefill_frequency": self._mean(prompt_iter_gaps)
                }

                json.dump(content, output_file, indent=2)