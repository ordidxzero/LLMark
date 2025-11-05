from llmark.vllm import VLLMBenchmarkRunner, VLLMBenchmarkRunnerV2, CommandTemplateV2, CommandArgs, NSYSOptions
from pathlib import Path
from tqdm import tqdm
import argparse, requests, json
from typing import Dict, Tuple, Callable, Optional

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

REGISTRY: Dict[Tuple[int, Optional[int]], Callable] = {}
def exp_register(episode: int, experiment: Optional[int] = None):
    def deco(fn: Callable):
        REGISTRY[(episode, experiment)] = fn
        return fn
    
    return deco

def get_nsys_options(args: argparse.Namespace) -> Optional[NSYSOptions]:
    if not args.enable_nsys:
        return None

    args_dict = vars(args)
    nsys_args = {}

    for k, v in args_dict.items():
        if k.startswith("nsys_"):
            k = k.replace("nsys_", "")
            nsys_args[k] = v

        if k == 'devices':
            nsys_args['cuda_visible_devices'] = v

    quantization_method = args.quantization if args.quantization is not None else "FP16"
    nsys_args['output'] = f'ep{args.ep}_tp_{args.nsys_tp}_pp_{args.nsys_pp}_rr_{args.nsys_request_rate}.nsys-rep'

    return NSYSOptions.from_json(nsys_args)

def get_quantization_config(args: argparse.Namespace):
    server_args = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "dtype": "bfloat16",
        "tp": 2,
        "pp": 2,
        "quantization": None
    }

    benchmark_args = {
        "model": "meta-llama/Llama-3.1-8B-Instruct"
    }
    if not args.quantization:
        return server_args.copy(), benchmark_args.copy()
    
    QUANTIZED_MODEL = {
        'awq': 'Llama-3.1-8B-Instruct-AWQ-Official-INT4',
        'awq_marlin': 'Llama-3.1-8B-Instruct-AWQ-Official-INT4',
        'gptq_exllamav2': 'Llama-3.1-8B-Instruct-GPTQ-ExLlamaV2-INT4',
        'gptq_marlin': 'Llama-3.1-8B-Instruct-GPTQ-Marlin-INT4',
        'smooth_quant': 'Llama-3.1-8B-Instruct-Smooth-Quant-INT8',
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
        },
        'smooth_quant': {
            'dtype': 'auto',
            "quantization": None,
        }
    }

    def add_path(model:str):
        return f"./quantized_model/{model}"

    def add_path_and_tokenizer(model: str):
        return f"./quantized_model/{model} --tokenizer ./quantized_model/{model}"
    
    model = QUANTIZED_MODEL[args.quantization]
    config = MODEL_CONFIGURATION[args.quantization]

    server_args["model"] = add_path(model)
    server_args.update(config)

    benchmark_args["model"] = add_path_and_tokenizer(model)

    return server_args, benchmark_args

@exp_register(episode=5, experiment=1)
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

@exp_register(episode=5, experiment=2)
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

@exp_register(episode=5, experiment=3)
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

@exp_register(episode=5, experiment=4)
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

@exp_register(episode=5, experiment=5)
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


@exp_register(episode=1)
def run_ep1(args: argparse.Namespace):
    cuda_visible_devices = args.devices
    
    DATASETS = [(2048, 2048), (128, 2048), (2048, 128), (128, 128)]
    pbar = tqdm(total=len(DATASETS))

    def rename(device_prefix: str, prefix: str, d: CommandArgs) -> str:
        input_len = d['input_len']
        output_len = d['output_len']
        filename = f'input_len_{input_len}_output_len_{output_len}'

        if prefix != '':
            filename = f'{prefix}_{filename}'
        else:
            filename = f'{device_prefix}_{filename}'

        return filename + '.log'

    SERVER_CMD = CommandTemplateV2('vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests', mapping=rename)
    BENCHMARK_CMD = CommandTemplateV2('vllm bench serve --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --ignore-eos --random-input-len {input_len} --random-output-len {output_len} --num-prompt 4096', mapping=rename)

    runner = VLLMBenchmarkRunnerV2(server_cmd=SERVER_CMD, benchmark_cmd=BENCHMARK_CMD, log_dir=Path(f"./output/vLLM/BF16"), envs={"CUDA_VISIBLE_DEVICES": cuda_visible_devices, "VLLM_USE_V1": '1'})

    for t in DATASETS:
        input_len, output_len = t
        runner.set_prefix(f'ep1_1st')
        runner.run(input_len=input_len, output_len=output_len)
        if pbar is not None:
            pbar.update(1)

@exp_register(episode=6)
def run_ep6(args: argparse.Namespace):
    cuda_visible_devices = args.devices
    disable_tqdm = args.disable_tqdm
    enforce_eager = args.enforce_eager
    nsys_options = get_nsys_options(args)
    server_args, benchmark_args = get_quantization_config(args)
    quantization_method = args.quantization if args.quantization is not None else "FP16"

    EXP_CONFIGURATION = [(256, 1), (256, 4), (256, 16), (256, 64), (1024, 256)]
    EXP_CONFIGURATION = [(32768, 64)]
    DATASETS = {
        "decode_heavy": (128, 2048),
    }
    
    
    def rename(device_prefix: str, prefix: str, d: CommandArgs) -> str:
        max_num_seqs = d['max_num_seqs']
        num_prompts = d['num_prompts']
        filename = f'max_num_seqs_{max_num_seqs}_num_prompts_{num_prompts}'

        if prefix != '':
            filename = f'{prefix}_{filename}'
        else:
            filename = f'{device_prefix}_{filename}'

        return filename + '.log'

    SERVER_CMD = CommandTemplateV2('vllm serve $model --dtype $dtype --quantization $quantization --pipeline-parallel-size $pp --tensor-parallel-size $tp --disable-log-requests --max-num-seqs {max_num_seqs} --enable-chunked-prefill False', partial_variables=server_args, mapping=rename)
    BENCHMARK_CMD = CommandTemplateV2('uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model $model --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --ignore-eos --hf-input-len {input_len} --hf-output-len {output_len} --hf_split 1k --num-prompt {num_prompts} --request-rate 10', partial_variables=benchmark_args, mapping=rename) # type: ignore

    SERVER_CMD.set_eager(enforce_eager)

    if nsys_options:
        SERVER_CMD.wrap_nsys(nsys_options)

    runner = VLLMBenchmarkRunnerV2(server_cmd=SERVER_CMD, benchmark_cmd=BENCHMARK_CMD, log_dir=Path(f"./output/vLLM/{quantization_method}"), envs={"CUDA_VISIBLE_DEVICES": cuda_visible_devices, "VLLM_ENGINE_ITERATION_TIMEOUT_S": '180'})

    if nsys_options is None:
        pbar = tqdm(total=len(DATASETS) * len(EXP_CONFIGURATION))
        if disable_tqdm:
            pbar = None

        for key, value in DATASETS.items():
            input_len, output_len = value
            runner.set_prefix(f'ep6_{key}')
            for num_prompts, max_num_seqs in EXP_CONFIGURATION:

                runner.run(key=key, max_num_seqs=max_num_seqs, num_prompts=num_prompts, input_len=input_len, output_len=output_len)
                if pbar is not None:
                    pbar.update(1)
        
        return
    
    key = nsys_options.workload
    input_len, output_len = DATASETS[key]
    max_num_seqs = nsys_options.max_num_seqs
    num_prompts = nsys_options.num_prompts

    runner.set_prefix(f"nsys_{key}")
    runner.run(max_num_seqs=max_num_seqs, num_prompts=num_prompts, input_len=input_len, output_len=output_len)

@exp_register(episode=9)
def run_ep9(args: argparse.Namespace):
    cuda_visible_devices = args.devices
    disable_tqdm = args.disable_tqdm
    enforce_eager = args.enforce_eager
    nsys_options = get_nsys_options(args)

    NUM_REQUESTS_AND_BATCH_SIZE = [(1024, 256), (256, 1), (256, 2), (256, 4), (256, 8), (256, 16), (256, 32), (512, 64), (1024, 128)]
    PARALLELISM_STRATEGIES = [(1, 4), (2, 2), (4, 1)] # TP and PP
    NUM_REQUESTS_AND_BATCH_SIZE = [(32768, 16)]
    PARALLELISM_STRATEGIES = [(1, 4)]
    REQUESTS_RATES = [1, 2, 4, 6, 8, 10, 12, 14, 16, float('inf')]
    REQUESTS_RATES = [1, 1, 1, 1]

    def rename(device_prefix: str, prefix: str, d: CommandArgs) -> str:
        tensor_parallelism = d['tp']
        pipeline_parallelism = d['pp']
        request_rate = d['request_rate']
        filename = f'tp_{tensor_parallelism}_pp_{pipeline_parallelism}_rr_{request_rate}'

        if prefix != '':
            filename = f'{prefix}_{filename}'
        else:
            filename = f'{device_prefix}_{filename}'

        return filename + '.log'


    SERVER_CMD = CommandTemplateV2('vllm serve codellama/CodeLlama-34b-hf --dtype bfloat16 --tensor-parallel-size {tp} --pipeline-parallel-size {pp} --disable-log-requests --max-num-seqs {max_num_seqs} --enable-chunked-prefill False', mapping=rename)
    BENCHMARK_CMD = CommandTemplateV2('uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model codellama/CodeLlama-34b-hf --dataset-name hf --dataset-path squeezebits/dynamic_sonnet_llama3 --ignore-eos --hf-input-len {input_len} --hf-output-len {output_len} --hf_split 1k --num-prompt {num_prompts} --request-rate {request_rate}', mapping=rename) # type: ignore

    SERVER_CMD.set_eager(enforce_eager)

    if nsys_options:
        SERVER_CMD.wrap_nsys(nsys_options)

    if nsys_options is None:
        pbar = tqdm(total=len(NUM_REQUESTS_AND_BATCH_SIZE) * len(PARALLELISM_STRATEGIES) * len(REQUESTS_RATES))
        if disable_tqdm:
            pbar = None

        input_len, output_len = (256, 256)

        for config in NUM_REQUESTS_AND_BATCH_SIZE:
            num_prompts, max_num_seqs = config
            for strategy in PARALLELISM_STRATEGIES:
                tensor_parallelism_size, pipeline_parallelism_size = strategy
                for request_rate in REQUESTS_RATES:
                    runner = VLLMBenchmarkRunnerV2(server_cmd=SERVER_CMD, benchmark_cmd=BENCHMARK_CMD, log_dir=Path(f"./output/vLLM/ep9/bs_{max_num_seqs}"), envs={"CUDA_VISIBLE_DEVICES": cuda_visible_devices, "VLLM_ENGINE_ITERATION_TIMEOUT_S": '180'})
                    runner.run(tp=tensor_parallelism_size, pp=pipeline_parallelism_size, max_num_seqs=max_num_seqs, input_len=input_len, output_len=output_len, num_prompts=num_prompts, request_rate=request_rate)

                    if pbar is not None:
                        pbar.update(1)
                        
        if pbar is not None:
            pbar.close()
        
        return
    
    tensor_parallelism_size = nsys_options.tp
    pipeline_parallelism_size = nsys_options.pp
    request_rate = nsys_options.request_rate
    input_len, output_len = (256, 256)
    max_num_seqs = 256
    num_prompts = nsys_options.num_prompts
    runner = VLLMBenchmarkRunnerV2(server_cmd=SERVER_CMD, benchmark_cmd=BENCHMARK_CMD, log_dir=Path(f"./output/vLLM/ep9/bs_{max_num_seqs}"), envs={"CUDA_VISIBLE_DEVICES": cuda_visible_devices, "VLLM_ENGINE_ITERATION_TIMEOUT_S": '600'})

    runner.set_prefix(f"nsys_")
    runner.run(tp=tensor_parallelism_size, pp=pipeline_parallelism_size, max_num_seqs=max_num_seqs, input_len=input_len, output_len=output_len, num_prompts=num_prompts, request_rate=request_rate)

@exp_register(episode=0)
def run_server(args: argparse.Namespace):
    cuda_visible_devices = args.devices

    cmd = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --disable-log-requests --max-num-seqs 256 --max-num-batched-tokens 8192 --max-model-len 8192"

    import os

    os.system(cmd)
    

def main(args: argparse.Namespace):
    episode = args.ep
    num_exp = args.exp

    key = (episode, num_exp)
    fn = REGISTRY.get(key)

    if not fn:
        available = ", ".join(f"(ep={e} & exp={x})" for (e, x) in sorted(REGISTRY))
        raise ValueError(f"Unknown combination ep={args.ep}, exp={args.exp}. Available: {available}")
    
    return fn(args)

def cmd_builder(args: argparse.Namespace):
    model = args.model
    dtype = args.dtype
    devices = args.devices
    quantization = args.quantization

    pass

def parse_args_v2():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", help="사용하고자하는 모델", choices=["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3.1-8B-Instruct"])
    
    subparsers = parser.add_subparsers(dest="action", help="Desired action to perform")

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--dtype", help="precision. quantization 옵션을 사용하는 경우 무시됨", choices=["float16", "bfloat16"], default="bfloat16")
    parent_parser.add_argument("--devices", help='CUDA_VISIBLE_DEVICES 값', default='0')
    parent_parser.add_argument("--quantization", help='Quantization Method', choices=['awq', 'awq_marlin', 'gptq_marlin', 'gptq_exllamav2', 'smooth_quant'])
    workload_group = parent_parser.add_argument_group("Workload")
    workload_group.add_argument("--dataset", choices=["sonnet", "random"])
    workload_group.add_argument("--workload", choices=["prefill-heavy", 'decode-heavy'], help="Preset workload name. Prefill-heavy: input 2048/output 128, Decode-heavy: input 128/output 2048")
    workload_group.add_argument("--input-len", type=int, help="Input sequence length", default=512)
    workload_group.add_argument("--output-len", type=int, help="Output sequence length", default=512)
    workload_group.add_argument("--num-prompts", type=int, help="Number of samples", default=512)
    workload_group.add_argument("--ignore-eos", action='store_true')
    workload_group.add_argument("--sonnet-split", choices=['1k', '2k', '4k', '8k'])
    workload_group.add_argument("--request-rate", type=float, help="Request rate", default=float("inf"))
    server_config_group = parent_parser.add_argument_group("Server Configuration")
    server_config_group.add_argument("--max-num-seqs", type=int)
    server_config_group.add_argument("--max-num-batched-tokens", type=int, default=16384)
    server_config_group.add_argument("--max-model-len", type=int, default=16384)
    server_config_group.add_argument("--enforce-eager", action='store_true')

    nsys_parser = subparsers.add_parser("nsys", help="Profiling with Nsight System", parents=[parent_parser])
    nsys_parser.add_argument("--delay", default=50, type=int)
    nsys_parser.add_argument("--duration", type=lambda x: x if x is None else int(x))
    bench_parser = subparsers.add_parser("bench", help="Benchmarking model", parents=[parent_parser])
    bench_parser.add_argument("--disable-tqdm", action='store_true')


    return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("model", help="사용하고자하는 모델")"
    # parser.add_argument("--dtype", help="precision. quantization 옵션을 사용하는 경우 무시됨", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--ep", help="SqueezeBits 포스팅 넘버", type=int, required=True)
    parser.add_argument("--exp", help="SqueezeBits 포스팅 내 실험 순서", type=lambda x: x if x is None else int(x))
    parser.add_argument("--devices", help='CUDA_VISIBLE_DEVICES 값', default='0')
    parser.add_argument("--quantization", help='Quantization Method', choices=['awq', 'awq_marlin', 'gptq_marlin', 'gptq_exllamav2', 'gptq_machete', 'smooth_quant'])
    parser.add_argument("--enforce-eager", action='store_true')
    parser.add_argument("--disable-tqdm", action='store_true')
    parser.set_defaults(enable_nsys=False)

    subparser = parser.add_subparsers()
    nsys_parser = subparser.add_parser("nsys")
    nsys_parser.set_defaults(enable_nsys=True, disable_tqdm=True)
    nsys_parser.add_argument("--delay", default=60, type=int, dest='nsys_delay')
    nsys_parser.add_argument("--duration", type=lambda x: x if x is None else int(x), dest='nsys_duration', default=argparse.SUPPRESS)
    nsys_parser.add_argument("--workload", choices=['prefill_heavy', 'decode_heavy'], default='prefill_heavy', dest='nsys_workload')
    nsys_parser.add_argument("--max-num-seqs", default=1, dest='nsys_max_num_seqs')
    nsys_parser.add_argument("--num-prompts", default=1024, dest='nsys_num_prompts')
    nsys_parser.add_argument("--tp", default=1, dest='nsys_tp')
    nsys_parser.add_argument("--pp", default=1, dest='nsys_pp')
    nsys_parser.add_argument("--request-rate", default=float('inf'), dest='nsys_request_rate')

    args =  parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # breakpoint()

    print(args)

    main(args)

    # 12.8.1-runtime-ubuntu22.04