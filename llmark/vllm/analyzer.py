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

        return iter_num, prompt_throughput, generation_throughput, prompt_seqs, generation_seqs, preemption_seqs, latency_iter
    
    def _get_prompt_iter_gap(self, iter_nums: List[int]):
        gaps = []
        for i in range(1, len(iter_nums)):
            prev_iter = iter_nums[i - 1]
            cur_iter = iter_nums[i]
            gaps.append(cur_iter - prev_iter)

        return gaps
    
    def _mean(self, d: List[int | float]):
        return round(sum(d) / len(d), 2)

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
                "preemption_seqs": []
            }
            with open(logfile_path, "r") as logfile:
                for line in logfile:
                    if "metrics.py" in line:
                        iter_num, prompt_throughput, generation_throughput, prompt_seqs, generation_seqs, preemption_seqs, latency = self._parse_line(line)
                        rest_state['preemption_seqs'].append(preemption_seqs)
                        
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

            if not dist_dir.exists():
                dist_dir.mkdir(parents=True)

            with open(dist_dir / f"{logfile_path.stem}.json", "w") as output_file:
                content = {
                    "avg_running_seqs": self._mean(all_running_seqs),
                    "avg_prompt_seqs": self._mean(prompt_state['running_seqs']),
                    "avg_generation_seqs": self._mean(generation_state['running_seqs']),
                    "avg_prompt_latency": self._mean(prompt_state['latencies']),
                    "avg_generation_latency": self._mean(generation_state['latencies']),
                    "prefill_frequency": self._mean(prompt_iter_gaps)
                }

                json.dump(content, output_file, indent=2)