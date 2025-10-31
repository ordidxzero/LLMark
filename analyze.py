from pathlib import Path
from llmark.vllm.analyzer import VLLMAnalyzer
from llmark.trtllm.analyzer import TensorRTLLMAnalyzer

methods = ['smooth_quant']

for m in methods:
    analyzer = VLLMAnalyzer(log_dir=Path(f"./output/vLLM/{m}/server"))

    analyzer.run("ep6_prefill_heavy_max_num_seqs_1_num_prompts_256.log", save_dir=Path(f"./output/final/{m}"))