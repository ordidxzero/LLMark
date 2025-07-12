from llmark.vllm import VLLMBenchmarkRunner
from pathlib import Path
from tqdm import tqdm

BENCHMARK_CMD = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 4096"
SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --disable-frontend-multiprocessing"
runner = VLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output"))

INPUT_LENGTHS = [2048, 128]
OUTPUT_LENGTHS = [2048, 128]
MAX_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
REQUEST_RATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

pbar = tqdm(total= len(OUTPUT_LENGTHS) * len(INPUT_LENGTHS))

for output_len in OUTPUT_LENGTHS:
    for input_len in INPUT_LENGTHS:
        runner.set_log_prefix("A100-80GB_exp1")
        runner.run(input_len=input_len, output_len=output_len)
        pbar.update(1)

BENCHMARK_CMD_2 = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 4096"
SERVER_CMD_2 = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --disable-frontend-multiprocessing --max-num-seqs {max_num_seqs}"
runner_2 = VLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD_2, server_cmd=SERVER_CMD_2, log_dir=Path("./output"))

# for max_num_seqs in MAX_BATCH_SIZES:
#     runner_2.set_log_prefix("A100-80GB_exp2")
#     runner_2.run(max_num_seqs=max_num_seqs, input_len=2048, output_len=128)
#     pbar.update(1)

# BENCHMARK_CMD_3 = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 512 --request-rate {request_rate}"
# SERVER_CMD_3 = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --disable-frontend-multiprocessing"
# runner_3 = VLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD_3, server_cmd=SERVER_CMD_3, log_dir=Path("./output"))

# for request_rate in REQUEST_RATES:
#     runner_3.set_log_prefix("A100-80GB_exp3")
#     runner_3.run(request_rate=request_rate, input_len=2048, output_len=128)
#     pbar.update(1)

# MAX_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# MAX_NUM_OF_TOKENS = [256, 512, 1024, 2048, 4096, 8192, 16384]
# BENCHMARK_CMD = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 1024"
# SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --disable-frontend-multiprocessing --max-num-seqs {max_num_seqs} --max-num-batched-tokens {max_num_batched_tokens}"
# runner = VLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output"))
# Decode-Heavy: 1024 samples, 128/768
# for max_num_seqs in MAX_BATCH_SIZES:
#     for max_num_batched_tokens in MAX_NUM_OF_TOKENS:
#         runner.set_log_prefix("A100-80GB_exp4_decode_heavy")
#         runner.run(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, input_len=128, output_len=768)

# Prefill-Heavy: 1024 samples, 768/128
# for max_num_seqs in MAX_BATCH_SIZES:
#     for max_num_batched_tokens in MAX_NUM_OF_TOKENS:
#         runner.set_log_prefix("A100-80GB_exp4_prefill_heavy")
#         runner.run(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, input_len=768, output_len=128)