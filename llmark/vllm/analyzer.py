import re, json
from pathlib import Path
from llmark.utils import LogAnalyzer
from typing import List, Tuple


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
        gpu_free_block = int(result.group(10))
        cpu_total_block = int(result.group(11))
        cpu_free_block = int(result.group(12))
        actual_num_batched_tokens = int(result.group(13))
        prompt_seqs = int(result.group(14))
        generation_seqs = int(result.group(15))
        preemption_seqs = int(result.group(16))
        generation_tokens = int(result.group(17))
        latency_iter = float(result.group(18))

        return iter_num, prompt_throughput, generation_throughput, gpu_kv_cache_usage, gpu_total_block, gpu_free_block, running_seqs, prompt_seqs, generation_seqs, preemption_seqs, latency_iter

    def _parse_profile_line(self, line: str):
        result = re.search(r'INFO                     profiler ] iter: (\d+), Attention: (\d+.\d+) ms, MLP: (\d+.\d+) ms, LM Head: (\d+.\d+) ms, Total: (\d+.\d+) ms, Stage: (\w+)', line)
        assert result is not None, "ParseError"

        iter_num = int(result.group(1))
        attn_time = float(result.group(2))
        mlp_time = float(result.group(3))
        lm_head_time = float(result.group(4))
        total_time = float(result.group(5))
        stage = result.group(6)

        return iter_num, attn_time, mlp_time, lm_head_time, total_time, stage

    
    def _get_prompt_iter_gap(self, iter_nums: List[int]):
        gaps = []
        for i in range(1, len(iter_nums)):
            prev_iter = iter_nums[i - 1]
            cur_iter = iter_nums[i]
            gaps.append(cur_iter - prev_iter)

        return gaps
    
    def _mean(self, d: List[int | float]):
        return round(sum(d) / len(d), 2)
    
    def plot_sequential_data(self, blocks: List[int], filename: str, target_path: Path, title: str = 'GPU KV Cache Usage', ylabel: str = 'Percentage', ylim: int = 100):
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

    def calc_prompt_groups(self, groups: List[List[Tuple[int, float]]]):
        if len(groups) == 0:
            return 0, 0, 0
        running_seqs = []
        latencies = []

        for group in groups:
            running_seqs_in_group, latencies_in_group = [list(l) for l in zip(*group)]
            running_seqs.append(sum(running_seqs_in_group))
            latencies.append(sum(latencies_in_group))

        
        return self._mean(running_seqs), self._mean(latencies), len(running_seqs)

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
                "original_running_seqs": [],
                "running_seqs": [],
                "preemption_seqs": [],
                "gpu_total_block": -1,
                "gpu_kv_cache_usage": [],
                "prompt_groups": []
            }
            profiling_state = {
                "prefill": {
                    "attn": [],
                    "mlp": [],
                    "lm_head": []
                },
                "decode": {
                    "attn": [],
                    "mlp": [],
                    "lm_head": []
                }
            }
            tmp_prompt = None
            is_prompt_group = False
            prompt_group = []
            with open(logfile_path, "r") as logfile:
                for line in logfile:
                    if "profiler ]" in line:
                        _, attn_time, mlp_time, lm_head_time, total_time, stage = self._parse_profile_line(line)

                        profiling_state[stage]['attn'].append(attn_time)
                        profiling_state[stage]['mlp'].append(mlp_time)
                        profiling_state[stage]['lm_head'].append(lm_head_time)
                        continue

                    if "metrics.py" in line:
                        iter_num, prompt_throughput, generation_throughput, gpu_kv_cache_usage, gpu_total_block, gpu_free_block, running_seqs, prompt_seqs, generation_seqs, preemption_seqs, latency = self._parse_iter_line(line)
                        rest_state['preemption_seqs'].append(preemption_seqs)
                        rest_state['gpu_kv_cache_usage'].append(gpu_kv_cache_usage)
                        rest_state['original_running_seqs'].append(running_seqs)

                        if rest_state['gpu_total_block'] == -1:
                            rest_state['gpu_total_block'] = gpu_total_block
                        
                        if prompt_throughput != 0.0:
                            # Prompt Stage
                            prompt_state["iter_nums"].append(iter_num)
                            prompt_state['running_seqs'].append(prompt_seqs)
                            rest_state['running_seqs'].append(prompt_seqs)
                            prompt_state['latencies'].append(latency)

                            if is_prompt_group:
                                if tmp_prompt:
                                    prompt_group.append(tmp_prompt)
                                    tmp_prompt = None
                                prompt_group.append((prompt_seqs, latency))

                            if not is_prompt_group:
                                tmp_prompt = (prompt_seqs, latency)
                                is_prompt_group = True

                        elif generation_throughput != 0.0:
                            # Generation Stage
                            generation_state["iter_nums"].append(iter_num)
                            generation_state['running_seqs'].append(generation_seqs)
                            rest_state['running_seqs'].append(generation_seqs)
                            generation_state['latencies'].append(latency)

                            if is_prompt_group:
                                if len(prompt_group) > 2:
                                    rest_state['prompt_groups'].append(prompt_group)
                                is_prompt_group = False
                                tmp_prompt = None
                                prompt_group = []

                        else:
                            continue

            all_running_seqs = generation_state['running_seqs'] + prompt_state['running_seqs']
            prompt_iter_gaps = self._get_prompt_iter_gap(prompt_state['iter_nums'])


            dist_dir = save_dir if save_dir is not None else self._log_dir

            self.plot_sequential_data(rest_state['gpu_kv_cache_usage'], logfile_path.with_suffix(".png").name, save_dir)
            self.plot_sequential_data(rest_state['running_seqs'], "running_seqs_" + logfile_path.with_suffix('.png').name, save_dir, title="Running Seqs", ylabel='Count', ylim=256)

            prompt_group_avg_seqs, prompt_group_avg_latency, prompt_group_num = self.calc_prompt_groups(rest_state['prompt_groups'])

            if not dist_dir.exists():
                dist_dir.mkdir(parents=True)

            with open(dist_dir / f"{logfile_path.stem}.json", "w") as output_file:
                content = {
                    "origin_running_seqs": self._mean(rest_state['original_running_seqs']),
                    "avg_running_seqs": self._mean(all_running_seqs),
                    "avg_prompt_seqs": self._mean(prompt_state['running_seqs']),
                    "avg_generation_seqs": self._mean(generation_state['running_seqs']),
                    "avg_prompt_latency": self._mean(prompt_state['latencies']),
                    "avg_generation_latency": self._mean(generation_state['latencies']),
                    "preemption_seqs": sum(rest_state['preemption_seqs']),
                    "gpu_total_block": rest_state['gpu_total_block'],
                    "prefill_frequency": self._mean(prompt_iter_gaps),
                    "total_prompt_latency": sum(prompt_state['latencies']),
                    "num_stalls_by_prompt": len(prompt_state['iter_nums']),
                    "prompt_groups": {
                        "avg_running_seqs": prompt_group_avg_seqs,
                        "avg_group_latency": prompt_group_avg_latency,
                        "group_num": prompt_group_num
                    },
                    "profiling": {
                        "prefill": {
                            "attn": self._mean(profiling_state["prefill"]['attn']),
                            "mlp": self._mean(profiling_state["prefill"]['mlp']),
                            "lm_head": self._mean(profiling_state["prefill"]['lm_head']),
                        },
                        "decode": {
                            "attn": self._mean(profiling_state["decode"]['attn']),
                            "mlp": self._mean(profiling_state["decode"]['mlp']),
                            "lm_head": self._mean(profiling_state["decode"]['lm_head']),
                        }
                    }
                }

                json.dump(content, output_file, indent=2)