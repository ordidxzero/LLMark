from llmark.vllm import VLLMBenchmarkRunner, VLLMBenchmarkRunnerV2, CommandTemplateV2, CommandArgs
from pathlib import Path
from tqdm import tqdm
import argparse, requests, json

SQUEEZEBITS_N1_EXP1 = False
SQUEEZEBITS_N1_EXP2 = False
SQUEEZEBITS_N1_EXP3 = False

SQUEEZEBITS_N2_EXP1 = False
SQUEEZEBITS_N2_EXP2 = False

SQUEEZEBITS_N5_EXP1 = False
SQUEEZEBITS_N5_EXP2 = False
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
    BENCHMARK_CMD = CommandTemplateV2(BENCHMARK_CMD)
    # SERVER_CMD = "vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384"
    SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager"
    SERVER_CMD = CommandTemplateV2(SERVER_CMD)

    runner = VLLMBenchmarkRunnerV2(
        benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output/vLLM"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "CUDA_VISIBLE_DEVICES": '0'}
    )

    for hf_split in SUBSETS:
        runner.set_prefix("v2_test")
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

def run_ep5_experiment1(args: argparse.Namespace):
    # 측정해야할 것
    # ep5_graph_dynamic
    # ep5_eager_dynamic (with nsys)
    # ep5_fixed_batch_dynamic (with nsys)
    cuda_visible_devices = args.devices
    disable_tqdm = args.disable_tqdm

    print("Start SQUEEZEBITS_N5_EXP1")
    SUBSETS = ['1k', '2k', '4k', '8k']
    SUBSETS = ['1k']
    SUBSETS = list(reversed(SUBSETS))

    BENCHMARK_CMD= CommandTemplateV2("uv run _vllm/benchmarks/benchmark_serving.py --backend openai-chat --endpoint /v1/chat/completions --model meta-llama/Llama-3.1-8B-Instruct --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --hf-output-len 1024 --num-prompt 1024 --hf-split {hf_split}")
    # SERVER_CMD = CommandTemplateV2("vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384")
    SERVER_CMD = CommandTemplateV2("nsys profile -t cuda,nvtx -o eager_dynamic_{hf_split}_report --force-overwrite=true --delay 40 --duration 300 vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager")

    runner = VLLMBenchmarkRunnerV2(
        benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, enable_nvtx=True, log_dir=Path("./output/vLLM"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "CUDA_VISIBLE_DEVICES": cuda_visible_devices}
    )

    pbar = tqdm(total=len(SUBSETS))
    if disable_tqdm:
        pbar = None

    for hf_split in SUBSETS[:1]:
        runner.set_prefix("ep5_eager_dynamic")
        runner.run(hf_split=hf_split)

        if pbar is not None:
            pbar.update(1)

def run_ep5_experiment2(args: argparse.Namespace):
    # 측정해야할 것
    # ep5_graph_fixed
    # ep5_eager_fixed (with nsys)
    # ep5_fixed_batch_fixed (with nsys)
    cuda_visible_devices = args.devices
    disable_tqdm = args.disable_tqdm

    print("Start SQUEEZEBITS_N5_EXP2")
    SUBSETS = ['1k']
    # SUBSETS = list(reversed(SUBSETS))

    dynamic_dataset = {
        "1k": {
            "input_len": 512,
            "output_len": 628,
            "max_num_seqs": 165,
        },
        "2k": {
            "input_len": 1017,
            "output_len": 957,
            "max_num_seqs": 95,
        },
        "4k": {
            "input_len": 3076,
            "output_len": 873,
            "max_num_seqs": 45,
        },
        "8k": {
            "input_len": 7154,
            "output_len": 905,
            "max_num_seqs": 20,
        }
    }

    BENCHMARK_CMD= CommandTemplateV2("uv run _vllm/benchmarks/benchmark_serving.py --backend openai-chat --endpoint /v1/chat/completions --model meta-llama/Llama-3.1-8B-Instruct --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --ignore-eos --hf-input-len {input_len} --hf-output-len {output_len} --hf_split {hf_split} --num-prompt 1024")
    # SERVER_CMD = CommandTemplateV2("vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs {max_num_seqs} --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager")
    SERVER_CMD = CommandTemplateV2("nsys profile -t cuda,nvtx -o eager_default_fixed_{hf_split}_report --force-overwrite=true --sample=none --cpuctxsw=none --delay 60 --duration 300 vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager")

    runner = VLLMBenchmarkRunnerV2(
        benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, enable_nvtx=True, log_dir=Path("./output/vLLM"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "CUDA_VISIBLE_DEVICES": cuda_visible_devices}
    )

    pbar = tqdm(total=len(SUBSETS))
    if disable_tqdm:
        pbar = None

    for hf_split in SUBSETS[:1]:
        runner.set_prefix("ep5_eager_default_fixed")
        input_len = dynamic_dataset[hf_split]['input_len']
        output_len = dynamic_dataset[hf_split]['output_len']
        max_num_seqs = dynamic_dataset[hf_split]['max_num_seqs']
        runner.run(hf_split=hf_split, input_len=input_len, output_len=output_len)

        if pbar is not None:
            pbar.update(1)

def run_ep5_experiment3(args: argparse.Namespace):
    cuda_visible_devices = args.devices
    disable_tqdm = args.disable_tqdm

    print("Start SQUEEZEBITS_N5_EXP3")
    SUBSETS = ['1k', '2k', '4k', '8k']
    SUBSETS = ['1k', '2k', '4k', '8k']
    SUBSETS = list(reversed(SUBSETS))

    dynamic_dataset = {
        "1k": {
            "input_len": 512,
            "output_len": 628,
        },
        "2k": {
            "input_len": 1017,
            "output_len": 957,
        },
        "4k": {
            "input_len": 3076,
            "output_len": 1024,
        },
        "8k": {
            "input_len": 7154,
            "output_len": 905,
        }
    }

    BENCHMARK_CMD= CommandTemplateV2("uv run _vllm/benchmarks/benchmark_serving.py --backend openai-chat --endpoint /v1/chat/completions --model meta-llama/Llama-3.1-8B-Instruct --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --ignore-eos --hf-input-len {input_len} --hf-output-len {output_len} --hf_split {hf_split} --num-prompt 128")
    # SERVER_CMD = CommandTemplateV2("vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs 16 --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager")
    SERVER_CMD = CommandTemplateV2("nsys profile -t cuda,nvtx -o strict_fixed_batch_fixed_{hf_split}_report --force-overwrite=true --sample=none --cpuctxsw=none --delay 60 --duration 2400 vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs 16 --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager")

    runner = VLLMBenchmarkRunnerV2(
        benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, enable_nvtx=True, log_dir=Path("./output/vLLM"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "CUDA_VISIBLE_DEVICES": cuda_visible_devices}
    )

    pbar = tqdm(total=len(SUBSETS))
    if disable_tqdm:
        pbar = None

    for hf_split in SUBSETS[:1]:
        runner.set_prefix("ep5_strict_fixed_batch_fixed")
        input_len = dynamic_dataset[hf_split]['input_len']
        output_len = dynamic_dataset[hf_split]['output_len']
        runner.run(hf_split=hf_split, input_len=input_len, output_len=output_len)

        if pbar is not None:
            pbar.update(1)

def run_ep5_experiment4(args: argparse.Namespace):
    cuda_visible_devices = args.devices
    disable_tqdm = args.disable_tqdm

    print("Start SQUEEZEBITS_N5_EXP4")
    MAX_BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256]

    BENCHMARK_CMD= CommandTemplateV2("uv run _vllm/benchmarks/benchmark_serving.py --backend openai-chat --endpoint /v1/chat/completions --model meta-llama/Llama-3.1-8B-Instruct --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --ignore-eos --hf-input-len 512 --hf-output-len 128 --hf_split 1k --num-prompt {num_prompt}")
    # SERVER_CMD = CommandTemplateV2("vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs {max_num_seqs} --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager")
    SERVER_CMD = CommandTemplateV2("nsys profile -t cuda,nvtx -o ffn_test_bs_{max_num_seqs}_report --force-overwrite=true --sample=none --cpuctxsw=none --delay 60 --duration 30 vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs {max_num_seqs} --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager")

    runner = VLLMBenchmarkRunnerV2(
        benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, enable_nvtx=True, log_dir=Path("./output/vLLM"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "CUDA_VISIBLE_DEVICES": cuda_visible_devices}
    )

    pbar = tqdm(total=len(MAX_BATCH_SIZES))
    if disable_tqdm:
        pbar = None

    for max_num_seqs in MAX_BATCH_SIZES[:1]:
        runner.set_prefix(f"ep5_ffn_test")
        num_prompt = 1024
        if max_num_seqs < 8:
            num_prompt = max_num_seqs * 4
        elif max_num_seqs < 128:
            num_prompt = max_num_seqs * 8
        else:
            pass
        runner.run(max_num_seqs=max_num_seqs, num_prompt=num_prompt)

        if pbar is not None:
            pbar.update(1)

def run_ep5_experiment5(args: argparse.Namespace):
    cuda_visible_devices = args.devices
    disable_tqdm = args.disable_tqdm

    print("Start SQUEEZEBITS_N5_EXP3")
    SUBSETS = ['1k', '2k', '4k', '8k']
    SUBSETS = ['1k', '2k', '4k', '8k']
    SUBSETS = list(reversed(SUBSETS))

    dynamic_dataset = {
        "1k": {
            "input_len": 512,
            "output_len": 768,
        },
        "2k": {
            "input_len": 1017,
            "output_len": 768,
        },
        "4k": {
            "input_len": 3076,
            "output_len": 768,
        },
        "8k": {
            "input_len": 7154,
            "output_len": 768,
        }
    }

    BENCHMARK_CMD= CommandTemplateV2("uv run _vllm/benchmarks/benchmark_serving.py --backend openai-chat --endpoint /v1/chat/completions --model meta-llama/Llama-3.1-8B-Instruct --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --ignore-eos --hf-input-len {input_len} --hf-output-len {output_len} --hf_split {hf_split} --num-prompt 1024")
    # SERVER_CMD = CommandTemplateV2("vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs 16 --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager")
    SERVER_CMD = CommandTemplateV2("nsys profile -t cuda,nvtx -o fixed_output_fixed_{hf_split}_report --force-overwrite=true --sample=none --cpuctxsw=none --delay 60 --duration 300 vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs 16 --max-num-batched-tokens 16384 --max-model-len 16384 --enforce-eager")

    runner = VLLMBenchmarkRunnerV2(
        benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, enable_nvtx=True, log_dir=Path("./output/vLLM"), envs={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "CUDA_VISIBLE_DEVICES": cuda_visible_devices}
    )

    pbar = tqdm(total=len(SUBSETS))
    if disable_tqdm:
        pbar = None

    for hf_split in SUBSETS[:1]:
        runner.set_prefix("ep5_fixed_output_fixed")
        input_len = dynamic_dataset[hf_split]['input_len']
        output_len = dynamic_dataset[hf_split]['output_len']
        runner.run(hf_split=hf_split, input_len=input_len, output_len=output_len)

        if pbar is not None:
            pbar.update(1)

def run_ep6_experiment1(args: argparse.Namespace):
    cuda_visible_devices = args.devices
    disable_tqdm = args.disable_tqdm
    quantization_method = args.quantization
    QUANTIZED_MODEL = {
        'awq': 'Llama-3.1-8B-Instruct-AWQ-Official-INT4',
        'awq_marlin': 'Llama-3.1-8B-Instruct-AWQ-Official-INT4',
        'gptq_exllamav2': 'Llama-3.1-8B-Instruct-GPTQ-ExLlamaV2-INT4',
        'gptq_marlin': 'Llama-3.1-8B-Instruct-GPTQ-Marlin-INT4',
    }

    MODEL_CONFIGURATION = {
        'awq': {
            "dtype": "float16",
            "quantization": "awq",
        },
        'awq_marlin': {
            'dtype': 'auto',
            "quantization": None,
        },
        'gptq_marlin': {
            'dtype': 'auto',
            'quantization': None,
        },
        'gptq_exllamav2': {
            'dtype': 'auto',
            "quantization": 'gptq',
        }
    }

    CONFIGURATION = [(256, 1), (256, 4), (256, 16), (256, 64), (1024, 256)]
    DATASETS = {
        "decode_heavy": (128, 2048),
        # "prefill_heavy": (2048, 128)
    }
    model = QUANTIZED_MODEL[quantization_method]
    partial_args = MODEL_CONFIGURATION[quantization_method]
    
    def args_filter(d: CommandArgs) -> CommandArgs:
        if 'model' in d:
            del d['model']

        if 'dtype' in d:
            del d['dtype']

        return d
    
    def rename(device_prefix: str, prefix: str, d: CommandArgs) -> str:
        quantization_method = d['quantization']
        max_num_seqs = d['max_num_seqs']
        num_prompts = d['num_prompts']
        filename = f'{quantization_method}_max_num_seqs_{max_num_seqs}_num_prompts_{num_prompts}'

        if prefix != '':
            filename = f'{device_prefix}_{prefix}_{filename}'
        else:
            filename = f'{device_prefix}_{filename}'

        return filename + '.log'

    
    # NSYS_QUANTIZED_SERVER_CMD = CommandTemplateV2('nsys profile -t cuda,nvtx,osrt --cuda-memory-usage=true -o ep6_awq_{key}_max_num_seqs_{max_num_seqs}_report --force-overwrite=true --sample=none --cpuctxsw=none --delay 50 vllm serve ./quantized_model/{model} --dtype $dtype --quantization $quantization --disable-log-requests --max-num-seqs {max_num_seqs} --enable-chunked-prefill False --enforce-eager', partial_variables=partial_args, mapping=name_mapping)
    QUANTIZED_BENCHMARK_CMD = CommandTemplateV2('uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model ./quantized_model/{model} --tokenizer ./quantized_model/{model} --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --ignore-eos --hf-input-len {input_len} --hf-output-len {output_len} --hf_split 1k --num-prompt {num_prompts}', filtering=args_filter, mapping=rename)
    NSYS_QUANTIZED_SERVER_CMD = CommandTemplateV2('nsys profile -t cuda,nvtx -o ep6_{key}_max_num_seqs_{max_num_seqs}_report.nsys-rep --force-overwrite=true --sample=none --cpuctxsw=none --delay 50 vllm serve ./quantized_model/{model} --dtype $dtype --quantization $quantization --disable-log-requests --max-num-seqs {max_num_seqs} --enable-chunked-prefill False --enforce-eager', partial_variables=partial_args, filtering=args_filter, mapping=rename)
    QUANTIZED_SERVER_CMD = CommandTemplateV2('vllm serve ./quantized_model/{model} --dtype $dtype --quantization $quantization --disable-log-requests --max-num-seqs {max_num_seqs} --enable-chunked-prefill False', partial_variables=partial_args, filtering=args_filter, mapping=rename)
    runner = VLLMBenchmarkRunnerV2(benchmark_cmd=QUANTIZED_BENCHMARK_CMD ,server_cmd=QUANTIZED_SERVER_CMD, log_dir=Path(f"./output/vLLM/{quantization_method}"), envs={"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    pbar = tqdm(total=len(DATASETS) * len(CONFIGURATION))
    if disable_tqdm:
        pbar = None

    # Quantized Model
    for key, value in DATASETS.items():
        input_len, output_len = value
        runner.set_prefix(f'ep6_{key}')
        for num_prompts, max_num_seqs in CONFIGURATION:
            if max_num_seqs != 4:
                continue

            runner.run(quantization=quantization_method, key=key, model=model, max_num_seqs=max_num_seqs, num_prompts=num_prompts, input_len=input_len, output_len=output_len)
            if pbar is not None:
                pbar.update(1)


def run_ep6_experiment2(args: argparse.Namespace):
    CONFIGURATION = list(reversed([(256, 1), (256, 4), (256, 16), (256, 64), (1024, 256)]))
    DATASETS = {
        # "prefill_heavy": (2048, 128),
        "decode_heavy": (128, 2048),
    }

    NSYS_CONFIG = {
        "decode_heavy": {
            "1": {
                "duration": 0,
            },
            "4": {
                "duration": 0,
            },
            "16": {
                "duration": 0,
            },
            "64": {
                "duration": 0,
            },
            "256": {
                "duration": 0,
            },
        },
        "prefill_heavy": {
            "1": {
                "duration": 660,
            },
            "4": {
                "duration": 210,
            },
            "16": {
                "duration": 120,
            },
            "64": {
                "duration": 120,
            },
            "256": {
                "duration": 270,
            },
        },
    }

    cuda_visible_devices = args.devices
    disable_tqdm = args.disable_tqdm

    def args_filter(d: CommandArgs):
        a = ['model', 'key', 'input_len', 'output_len']
        
        for k in a:
            if k in d:
                del d[k]

        return d
    

    FP16_SERVER_CMD = CommandTemplateV2('vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype float16 --disable-log-requests --max-num-seqs {max_num_seqs} --enable-chunked-prefill False --enforce-eager', filtering=args_filter)
    FP16_SERVER_CMD = CommandTemplateV2('vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype float16 --disable-log-requests --max-num-seqs {max_num_seqs} --enable-chunked-prefill False', filtering=args_filter)
    NSYS_FP16_SERVER_CMD = CommandTemplateV2('nsys profile -t cuda,nvtx -o ep6_default_{key}_max_num_seqs_{max_num_seqs}_report.nsys-rep --force-overwrite=true --sample=none --cpuctxsw=none --delay 50 vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype float16 --disable-log-requests --max-num-seqs {max_num_seqs} --enable-chunked-prefill False --enforce-eager', filtering=args_filter)
    NSYS_FP16_SERVER_CMD = CommandTemplateV2('vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype float16 --disable-log-requests --max-num-seqs {max_num_seqs} --enable-chunked-prefill False --enforce-eager', filtering=args_filter)
    BENCHMARK_CMD = CommandTemplateV2('uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Llama-3.1-8B-Instruct --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --ignore-eos --hf-input-len {input_len} --hf-output-len {output_len} --hf_split 1k --num-prompt {num_prompts}', filtering=args_filter)
    
    runner = VLLMBenchmarkRunnerV2(benchmark_cmd=BENCHMARK_CMD, server_cmd=FP16_SERVER_CMD, log_dir=Path("./output/vLLM/FP16"), envs={"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    pbar = tqdm(total=len(DATASETS) * len(CONFIGURATION))
    if disable_tqdm:
        pbar = None

    for key, value in DATASETS.items():
        input_len, output_len = value
        runner.set_prefix(f'ep6_{key}_default')
        for num_prompts, max_num_seqs in CONFIGURATION:

            if max_num_seqs == 256:
                continue

            runner.run(model='meta-llama/Llama-3.1-8B-Instruct',key=key, max_num_seqs=max_num_seqs, num_prompts=num_prompts, input_len=input_len, output_len=output_len)

            if pbar is not None:
                pbar.update(1)

def main(args: argparse.Namespace):
    episode = args.ep
    num_exp = args.exp

    if episode == '5':
        if num_exp == '1':
            run_ep5_experiment1(args)
        if num_exp == '2':
            run_ep5_experiment2(args)
        if num_exp == '3':
            run_ep5_experiment3(args)
        if num_exp == '4':
            run_ep5_experiment4(args)
        if num_exp == '5':
            run_ep5_experiment5(args)

    if episode == '6':
        if num_exp == '1':
            args.quantization = 'awq'
            run_ep6_experiment1(args)
            args.quantization = 'awq_marlin'
            run_ep6_experiment1(args)
            args.quantization = 'gptq_exllamav2'
            run_ep6_experiment1(args)
            args.quantization = 'gptq_marlin'
            run_ep6_experiment1(args)
            run_ep6_experiment2(args)
        if num_exp == '2':
            run_ep6_experiment2(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ep", help="SqueezeBits 포스팅 넘버", required=True)
    parser.add_argument("--exp", help="SqueezeBits 포스팅 내 실험 순서", required=True)
    parser.add_argument("--devices", help='CUDA_VISIBLE_DEVICES 값', default='0')
    parser.add_argument("--quantization", help='Quantization Method', choices=['awq', 'awq_marlin', 'gptq_marlin', 'gptq_exllamav2'])
    parser.add_argument("--disable-tqdm", action='store_true')

    args = parser.parse_args()

    main(args)

    # 12.8.1-runtime-ubuntu22.04