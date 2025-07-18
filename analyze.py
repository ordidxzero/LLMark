from pathlib import Path
from llmark.vllm.analyzer import VLLMAnalyzer

analyzer = VLLMAnalyzer(log_dir=Path("./output/server"))

analyzer.run("A100-40GB_exp4_prefill_heacy_max_num_seqs_256*.log", save_dir=Path("./output/final"))