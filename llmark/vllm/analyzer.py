import re, json
from pathlib import Path
from llmark.utils import LogAnalyzer
from typing import List


class VLLMAnalyzer(LogAnalyzer):
    def __init__(self, log_dir: Path):
        super().__init__(log_dir)
    
    def _parse_line(self, line: str):
        result = re.search(r'iter: (\d+) Avg prompt throughput: (\d+.\d+) tokens/s, Avg generation throughput: (\d+.\d+) tokens/s, Running: (\d+) reqs, Swapped: (\d+) reqs, Pending: (\d+) reqs, GPU KV cache usage: (\d+.\d+)%, CPU KV cache usage: (\d+.\d+)%.Prompt stage: (-?\d+) reqs, Generation stage: (-?\d+) reqs, Preemption: (\d+) reqsLatenyMS: (\d+.\d+)', line)
        if result is None:
            breakpoint()
        
        iter_num = int(result.group(1))
        prompt_throughput = float(result.group(2))
        generation_throughput = float(result.group(3))
        running_seqs = int(result.group(4))
        # swapped_seqs = int(result.group(5))
        # pending_seqs = int(result.group(6))
        prompt_seqs = int(result.group(9))
        generation_seqs = int(result.group(10))
        preemption_seqs = int(result.group(11))
        latency_iter = float(result.group(12))

        return iter_num, prompt_throughput, generation_throughput, running_seqs, prompt_seqs, generation_seqs, preemption_seqs, latency_iter
    
    def _get_prompt_iter_gap(self, iter_nums: List[int]):
        gaps = []
        for i in range(1, len(iter_nums)):
            prev_iter = iter_nums[i - 1]
            cur_iter = iter_nums[i]
            gaps.append(cur_iter - prev_iter)

        return gaps

    def run(self, pattern: str, save_dir: Path = None):
        assert pattern is not None and pattern != "" and isinstance(pattern, str), "Error"
        
        for logfile_path in self._log_dir.glob(pattern):
            prompt_state = {
                "iter_nums": [],
                "running_seqs": [],
                "latencies": [],
                "group_latencies": [],
                "buffer": [],
                "group_buffer": []
            }
            prompt_fsm = {
                "flag": 0, # 0이면 초기화 상태, 1이면 prompt start 
                "type": 0 #  0이면 초기화 상태, 1이면 Prompt 한번, 2이면 Prompt 연속적으로 여러번
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
                        iter_num, prompt_throughput, generation_throughput, running_seqs, prompt_seqs, generation_seqs, preemption_seqs, latency = self._parse_line(line)
                        rest_state['preemption_seqs'].append(preemption_seqs)
                        if prompt_throughput == 0.0:
                            if prompt_seqs == 0 and prompt_fsm['flag'] == 1:
                                # Prompt stage가 끝난 직후
                                if prompt_fsm['type'] == 1:
                                    buffer = prompt_state['buffer']
                                    prompt_state['iter_nums'].append(buffer[0])
                                    prompt_state['running_seqs'].append(buffer[1])
                                    prompt_state['latencies'].append(buffer[2])
                                elif prompt_fsm['type'] == 2:
                                    group_buffer = prompt_state['group_buffer']
                                    prompt_state['group_latencies'].append(sum(group_buffer))
                                    prompt_state['iter_nums'].append(iter_num - int(group_buffer / 2))

                                prompt_fsm = {"flag": 0, "type": 0}
                                prompt_state["buffer"] = []
                                prompt_state['group_buffer'] = []
                            
                            if prompt_seqs != 0:
                                prompt_fsm['flag'] = 1
                            
                        if prompt_throughput != 0.0 and prompt_fsm['flag'] == 1:
                            prompt_state['group_buffer'].append(latency)
                            prompt_state['running_seqs'].append(prompt_seqs)
                            if prompt_fsm['type'] == 0:
                                prompt_state['buffer'] = [iter_num, prompt_seqs, latency]
                                prompt_fsm['type'] == 1
                            elif prompt_fsm['type'] == 1:
                                prompt_fsm['type'] == 2
                            else:
                                continue
                        
                        if prompt_throughput == 0.0 and prompt_fsm['flag'] == 0:
                            generation_state['iter_nums'].append(iter_num)
                            generation_state['running_seqs'].append(generation_seqs)
                            generation_state['latencies'].append(latency)
            
            all_running_seqs = generation_state['running_seqs'] + prompt_state['running_seqs']
            prompt_iter_gaps = self._get_prompt_iter_gap(prompt_state['iter_nums'])
            prompt_seqs_list = prompt_state['running_seqs']
            prompt_latencies = prompt_state['latencies']
            generation_seqs_list = generation_state['running_seqs']
            generation_latencies = generation_state['latencies']
            prompt_group_latencies = prompt_state['group_latencies']

            if len(prompt_latencies) == 0:
                breakpoint()


            dist_dir = save_dir if save_dir is not None else self._log_dir

            if not dist_dir.exists():
                dist_dir.mkdir(parents=True)

            with open(dist_dir / f"{logfile_path.stem}.json", "w") as output_file:
                content = {
                    "avg_running_seqs": round(sum(all_running_seqs) / len(all_running_seqs), 2),
                    "avg_prompt_seqs": round(sum(prompt_seqs_list) / len(prompt_seqs_list), 2),
                    "avg_generation_seqs": round(sum(generation_seqs_list) / len(generation_seqs_list), 2),
                    "avg_prompt_latency": round(sum(prompt_latencies) / len(prompt_latencies), 2),
                    "avg_generation_latency": round(sum(generation_latencies) / len(generation_latencies), 2),
                    "avg_prompt_group_latency": round(sum(prompt_group_latencies) / len(prompt_group_latencies), 2),
                    "prefill_frequency": round(sum(prompt_iter_gaps) / len(prompt_iter_gaps), 2)
                }

                json.dump(content, output_file, indent=2)