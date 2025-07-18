from llmark.vllm import VLLMBenchmarkRunner
from pathlib import Path
from tqdm import tqdm

# benchmark_cmd = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 4096"
# server_cmd = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests"
# runner = VLLMBenchmarkRunner(benchmark_cmd=benchmark_cmd, server_cmd=server_cmd, log_dir=Path("./output"))
# for output_len in [2048, 128]:
#     for input_len in [2048, 128]:
#         runner.set_log_prefix("A100-80GB_exp1")
#         runner.run(input_len=input_len, output_len=output_len)

# benchmark_cmd = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 4096"
# server_cmd = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs {max_num_seqs}"
# runner = VLLMBenchmarkRunner(benchmark_cmd=benchmark_cmd, server_cmd=server_cmd, log_dir=Path("./output"))
# for max_num_seqs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
#     runner.set_log_prefix("A100-80GB_exp2")
#     runner.run(max_num_seqs=max_num_seqs, input_len=2048, output_len=128)

# benchmark_cmd = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 512 --request-rate {request_rate}"
# server_cmd = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests"
# runner = VLLMBenchmarkRunner(benchmark_cmd=benchmark_cmd, server_cmd=server_cmd, log_dir=Path("./output"))
# for request_rate in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
#     runner.set_log_prefix("A100-80GB_exp3")
#     runner.run(request_rate=request_rate, input_len=2048, output_len=128)


MAX_BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256, 512]
MAX_NUM_OF_TOKENS = [1024, 2048, 4096, 8192, 16384]

BENCHMARK_CMD= "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 1024"
SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs {max_num_seqs} --max-num-batched-tokens {max_num_batched_tokens} --max-model-len 1024"
runner = VLLMBenchmarkRunner(
    benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
)

# Decode-Heavy
for max_num_seqs in MAX_BATCH_SIZES:
    for max_num_batched_tokens in MAX_NUM_OF_TOKENS:
        runner.set_log_prefix("A100-80GB_exp4_decode_heavy")
        runner.run(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, input_len=128, output_len=768)

# Prefill-Heavy
for max_num_seqs in MAX_BATCH_SIZES:
    for max_num_batched_tokens in MAX_NUM_OF_TOKENS:
        runner.set_log_prefix("A100-80GB_exp4_prefill_heavy")
        runner.run(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, input_len=768, output_len=128)