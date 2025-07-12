from llmark.vllm import VLLMBenchmarkRunner
from pathlib import Path
from tqdm import tqdm

BENCHMARK_CMD = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 4096"
SERVER_CMD = (
    "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests"
)
runner = VLLMBenchmarkRunner(
    benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output")
)

INPUT_LENGTHS = [2048, 128]
OUTPUT_LENGTHS = [2048, 128]
MAX_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
REQUEST_RATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

pbar = tqdm(
    total=len(OUTPUT_LENGTHS) * len(INPUT_LENGTHS)
    + len(MAX_BATCH_SIZES)
    + len(REQUEST_RATES)
)

for output_len in OUTPUT_LENGTHS:
    for input_len in INPUT_LENGTHS:
        runner.set_log_prefix("A100-40GB_exp1")
        runner.run(input_len=input_len, output_len=output_len)
        pbar.update(1)

BENCHMARK_CMD_2 = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 4096"
SERVER_CMD_2 = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs {max_num_seqs}"
runner_2 = VLLMBenchmarkRunner(
    benchmark_cmd=BENCHMARK_CMD_2, server_cmd=SERVER_CMD_2, log_dir=Path("./output")
)

for max_num_seqs in MAX_BATCH_SIZES:
    runner_2.set_log_prefix("A100-40GB_exp2")
    runner_2.run(max_num_seqs=max_num_seqs, input_len=2048, output_len=128)
    pbar.update(1)

BENCHMARK_CMD_3 = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 512 --request-rate {request_rate}"
SERVER_CMD_3 = (
    "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests"
)
runner_3 = VLLMBenchmarkRunner(
    benchmark_cmd=BENCHMARK_CMD_3, server_cmd=SERVER_CMD_3, log_dir=Path("./output")
)

for request_rate in REQUEST_RATES:
    runner_3.set_log_prefix("A100-40GB_exp3")
    runner_3.run(request_rate=request_rate, input_len=2048, output_len=128)
    pbar.update(1)
