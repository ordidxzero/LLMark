from llmark.vllm import VLLMBenchmarkRunner
from pathlib import Path
from tqdm import tqdm


MAX_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
MAX_NUM_OF_TOKENS = [256, 512, 1024, 2048, 4096, 8192, 16384]

BENCHMARK_CMD= "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 1024"
SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs {max_num_seqs}"
runner = VLLMBenchmarkRunner(
    benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
)

# Decode-Heavy
for max_num_seqs in MAX_BATCH_SIZES:
    for max_num_batched_tokens in MAX_NUM_OF_TOKENS:
        runner.set_log_prefix("A100-40GB_exp4_decode_heavy")
        runner.run(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, input_len=128, output_len=768)

# Prefill-Heavy
for max_num_seqs in MAX_BATCH_SIZES:
    for max_num_batched_tokens in MAX_NUM_OF_TOKENS:
        runner.set_log_prefix("A100-40GB_exp4_prefill_heavy")
        runner.run(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, input_len=768, output_len=128)