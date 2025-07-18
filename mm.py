from llmark.vllm import VLLMBenchmarkRunner
from pathlib import Path

benchmark_cmd = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --random-output-len {output_len} --ignore-eos --num-prompts 4096"
server_cmd = (
    "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests"
)
runner = VLLMBenchmarkRunner(
    benchmark_cmd=benchmark_cmd,
    server_cmd=server_cmd,
    log_dir=Path("./output"),
    envs={"CUDA_VISIBLE_DEVICES": "2"},
)
runner.set_log_prefix("testtest")
runner.run(input_len=128, output_len=100)
