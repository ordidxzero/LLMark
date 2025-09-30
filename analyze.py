from pathlib import Path
from llmark.vllm.analyzer import VLLMAnalyzer
from llmark.trtllm.analyzer import TensorRTLLMAnalyzer

methods = ['awq', 'awq_marlin', 'FP16', 'gptq_exllamav2', 'gptq_marlin']

for m in methods:
    analyzer = VLLMAnalyzer(log_dir=Path(f"./output/vLLM/{m}/server"))

    analyzer.run("A100_80GB_PCIe_ep6_prefill*.log", save_dir=Path(f"./output/final/{m}"))