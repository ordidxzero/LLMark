from pathlib import Path
from llmark.vllm.analyzer import VLLMAnalyzer
from llmark.trtllm.analyzer import TensorRTLLMAnalyzer

analyzer = VLLMAnalyzer(log_dir=Path("./output/vLLM/server"))

analyzer.run("A100_80GB_PCIe_n5_profiling_*.log", save_dir=Path("./output/vLLM/final"))