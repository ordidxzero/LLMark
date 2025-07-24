from pathlib import Path
from llmark.vllm.analyzer import VLLMAnalyzer

analyzer = VLLMAnalyzer(log_dir=Path("./output/vLLM/server"))

analyzer.run("A100-SXM4-40GB_n5_exp1_hf_split_8k_.log", save_dir=Path("./output/vLLM/final"))
