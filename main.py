from llmark.vllm import VLLMBenchmarkRunner
from pathlib import Path
from tqdm import tqdm

SQUEEZEBITS_N1_EXP1 = False
SQUEEZEBITS_N1_EXP2 = False
SQUEEZEBITS_N1_EXP3 = False

SQUEEZEBITS_N2_EXP1 = False
SQUEEZEBITS_N2_EXP2 = False

SQUEEZEBITS_N5_EXP1 = False
SQUEEZEBITS_N5_EXP2 = True
SQUEEZEBITS_N5_EXP1_1 = False
SQUEEZEBITS_N5_EXP2_1 = False

if SQUEEZEBITS_N1_EXP1:
    benchmark_cmd = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 4096"
    server_cmd = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests"
    runner = VLLMBenchmarkRunner(benchmark_cmd=benchmark_cmd, server_cmd=server_cmd, log_dir=Path("./output/vLLM"))
    for output_len in [2048, 128]:
        for input_len in [2048, 128]:
            runner.set_log_prefix("n1_exp1")
            runner.run(input_len=input_len, output_len=output_len)

if SQUEEZEBITS_N1_EXP2:
    benchmark_cmd = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len 2048 --ignore-eos --random-output-len 128 --num-prompts 4096"
    server_cmd = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs {max_num_seqs}"
    runner = VLLMBenchmarkRunner(benchmark_cmd=benchmark_cmd, server_cmd=server_cmd, log_dir=Path("./output/vLLM"))
    for max_num_seqs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        runner.set_log_prefix("n1_exp2")
        runner.run(max_num_seqs=max_num_seqs)

if SQUEEZEBITS_N1_EXP3:
    benchmark_cmd = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len 2048 --ignore-eos --random-output-len 128 --num-prompts 512 --request-rate {request_rate}"
    server_cmd = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests"
    runner = VLLMBenchmarkRunner(benchmark_cmd=benchmark_cmd, server_cmd=server_cmd, log_dir=Path("./output/vLLM"))
    for request_rate in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        runner.set_log_prefix("n1_exp3")
        runner.run(request_rate=request_rate, input_len=2048, output_len=128)

if SQUEEZEBITS_N2_EXP1:
    MAX_BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256, 512]
    MAX_NUM_OF_TOKENS = [1024, 2048, 4096, 8192, 16384]

    BENCHMARK_CMD= "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len 128 --ignore-eos --random-output-len 768 --num-prompts 1024"
    SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs {max_num_seqs} --max-num-batched-tokens {max_num_batched_tokens} --max-model-len 1024"
    runner = VLLMBenchmarkRunner(
        benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output/vLLM"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
    )

    # Decode-Heavy
    for max_num_seqs in MAX_BATCH_SIZES:
        for max_num_batched_tokens in MAX_NUM_OF_TOKENS:
            runner.set_log_prefix("n2_exp1_decode_heavy")
            runner.run(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens)

if SQUEEZEBITS_N2_EXP2:
    MAX_BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256, 512]
    MAX_NUM_OF_TOKENS = [1024, 2048, 4096, 8192, 16384]

    BENCHMARK_CMD= "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len 768 --ignore-eos --random-output-len 128 --num-prompts 1024"
    SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs {max_num_seqs} --max-num-batched-tokens {max_num_batched_tokens} --max-model-len 1024"
    runner = VLLMBenchmarkRunner(
        benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output/vLLM"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
    )

    # Prefill-Heavy
    for max_num_seqs in MAX_BATCH_SIZES:
        for max_num_batched_tokens in MAX_NUM_OF_TOKENS:
            runner.set_log_prefix("n2_exp1_prefill_heavy")
            runner.run(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens)

if SQUEEZEBITS_N5_EXP1:
    print("Start SQUEEZEBITS_N5_EXP1")
    SUBSETS = ['1k', '2k', '4k', '8k']
    SUBSETS = reversed(SUBSETS)

    BENCHMARK_CMD= "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --hf-output-len 1024 --num-prompt 1024 --hf-split {hf_split}"
    # SERVER_CMD = "vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384"
    SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager"

    runner = VLLMBenchmarkRunner(
        benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output/vLLM"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "CUDA_VISIBLE_DEVICES": '0'}
    )

    for hf_split in SUBSETS:
        runner.set_log_prefix("n5_profiling_dynamic")
        runner.run(hf_split=hf_split)

if SQUEEZEBITS_N5_EXP1_1:
    print("Start SQUEEZEBITS_N5_EXP1_1")
    SUBSETS = ['4k']
    SUBSETS = reversed(SUBSETS)

    BENCHMARK_CMD= "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --hf-output-len 896 --num-prompt 1024 --hf-split {hf_split}"
    # SERVER_CMD = "vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384"
    SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager"

    runner = VLLMBenchmarkRunner(
        benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output/vLLM"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "CUDA_VISIBLE_DEVICES": '0'}
    )

    for hf_split in SUBSETS:
        runner.set_log_prefix("n5_profile_dynamic")
        runner.run(hf_split=hf_split)

if SQUEEZEBITS_N5_EXP2:
    print("Start SQUEEZEBITS_N5_EXP2")
    SUBSETS = ['1k', '2k', '4k', '8k']
    SUBSETS = list(reversed(SUBSETS))
    dynamic_dataset = {
        "1k": {
            "input_len": 503,
            "output_len": 828,
        },
        "2k": {
            "input_len": 1008,
            "output_len": 843,
        },
        "4k": {
            "input_len": 3067,
            "output_len": 858,
        },
        "8k": {
            "input_len": 7145,
            "output_len": 803,
        }
    }

    BENCHMARK_CMD= "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --ignore-eos --hf-input-len {input_len} --hf-output-len {output_len} --hf_split {hf_split} --num-prompt 1024"
    # BENCHMARK_CMD= "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --ignore-eos --random-input-len {input_len} --random-output-len {output_len} --num-prompt 1024"
    # SERVER_CMD = "vllm serve --model meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384"
    SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager"

    runner = VLLMBenchmarkRunner(
        benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output/vLLM"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "CUDA_VISIBLE_DEVICES": '0'}
    )

    for hf_split in SUBSETS:
        runner.set_log_prefix("n5_profiling_fixed")
        input_len = dynamic_dataset[hf_split]['input_len']
        output_len = dynamic_dataset[hf_split]['output_len']
        runner.run(hf_split=hf_split, input_len=input_len, output_len=output_len)

if SQUEEZEBITS_N5_EXP2_1:
    print("Start SQUEEZEBITS_N5_EXP2_1")
    SUBSETS = ['2k']
    SUBSETS = list(reversed(SUBSETS))
    dynamic_dataset = {
        "1k": {
            "input_len": 503,
            "output_len": 828,
        },
        "2k": {
            "input_len": 1008,
            "output_len": 600,
        },
        "4k": {
            "input_len": 3067,
            "output_len": 858,
        },
        "8k": {
            "input_len": 7145,
            "output_len": 801,
        }
    }

    BENCHMARK_CMD= "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --ignore-eos --hf-input-len {input_len} --hf-output-len {output_len} --hf_split {hf_split} --num-prompt 1024"
    # BENCHMARK_CMD= "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --ignore-eos --random-input-len {input_len} --random-output-len {output_len} --num-prompt 1024"
    # SERVER_CMD = "vllm serve --model meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384"
    SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384"

    runner = VLLMBenchmarkRunner(
        benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output/vLLM"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "CUDA_VISIBLE_DEVICES": '2'}
    )

    for hf_split in SUBSETS:
        runner.set_log_prefix("n5_additional_fixed")
        input_len = dynamic_dataset[hf_split]['input_len']
        output_len = dynamic_dataset[hf_split]['output_len']
        runner.run(hf_split=hf_split, input_len=input_len, output_len=output_len)