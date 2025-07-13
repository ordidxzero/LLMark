from pathlib import Path
from llmark.vllm.analyzer import VLLMAnalyzer

analyzer = VLLMAnalyzer(log_dir=Path("./output/server"))

analyzer.run("A100-40GB_exp1_*.log", save_dir=Path("./output/final"))