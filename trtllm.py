from llmark.trtllm import TensorRTLLMBenchmarkRunner, TensorRTLLMDatasetGenerator
from pathlib import Path
from tqdm import tqdm

SQUEEZEBITS_N1_EXP1 = False
SQUEEZEBITS_N1_EXP2 = False
SQUEEZEBITS_N1_EXP3 = False

SQUEEZEBITS_N2_EXP1 = False
SQUEEZEBITS_N2_EXP2 = False

SQUEEZEBITS_N5_EXP1 = True
SQUEEZEBITS_N5_EXP2 = False

if SQUEEZEBITS_N1_EXP1:
    BUILD_CMD = "trtllm-bench --model meta-llama/Meta-Llama-3-8B build --max_batch_size 256 --max_num_tokens 8192 --dataset {build_dataset_path}"
    BENCHMARK_CMD = "/app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark --engine_dir /tmp/meta-llama/Meta-Llama-3-8B/tp_1_pp_1/ --dataset {benchmark_dataset_path} --log_iteration_data --streaming true"

    runner = TensorRTLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD, build_cmd=BUILD_CMD, log_dir=Path("/code/output/TensorRT-LLM"), envs={"CUDA_VISIBLE_DEVICES": "3"})

    generator = TensorRTLLMDatasetGenerator(script_path=Path("/app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py"), save_dir=Path("/code/_TensorRT-LLM/datasets"))

    OUTPUT_LENGTHS = [2048, 128]
    INPUT_LENGTHS = [2048, 128]

    pbar = tqdm(total = 4)

    for output_len in OUTPUT_LENGTHS:
        for input_len in INPUT_LENGTHS:
            filename = f"{input_len}x{output_len}_sample_4096"
            build_dataset_path = generator.generate(filename, "meta-llama/Meta-Llama-3-8B", input_len=input_len, output_len=output_len, num_requests=4096, for_build=True)
            benchmark_dataset_path = generator.generate(filename, "meta-llama/Meta-Llama-3-8B", input_len=input_len, output_len=output_len, num_requests=4096)

            runner.set_log_prefix("exp1")
            runner.init(build_dataset_path=build_dataset_path, benchmark_dataset_path=benchmark_dataset_path, input_len=input_len, output_len=output_len)

            runner.build_model()

            runner.run_benchmark()

            pbar.update(1)

if SQUEEZEBITS_N1_EXP2:
    BUILD_CMD = "trtllm-bench --model meta-llama/Meta-Llama-3-8B build --max_batch_size {max_batch_size} --max_num_tokens 8192 --dataset {build_dataset_path}"
    BENCHMARK_CMD = "/app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark --engine_dir /tmp/meta-llama/Meta-Llama-3-8B/tp_1_pp_1/ --dataset {benchmark_dataset_path} --log_iteration_data --streaming true"
    runner = TensorRTLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD, build_cmd=BUILD_CMD, log_dir=Path("/code/output/TensorRT-LLM"), envs={"CUDA_VISIBLE_DEVICES": "3"})

    generator = TensorRTLLMDatasetGenerator(script_path=Path("/app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py"), save_dir=Path("/code/_TensorRT-LLM/datasets"))
    MAX_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    pbar = tqdm(total = len(MAX_BATCH_SIZES))
    for max_batch_size in MAX_BATCH_SIZES:
        filename = "2048x128_sample_4096"
        build_dataset_path = generator.generate(filename, "meta-llama/Meta-Llama-3-8B", input_len=2048, output_len=128, num_requests=4096, for_build=True)
        benchmark_dataset_path = generator.generate(filename, "meta-llama/Meta-Llama-3-8B", input_len=2048, output_len=128, num_requests=4096)
        runner.set_log_prefix("n1_exp2")
        runner.run(build_dataset_path=build_dataset_path, benchmark_dataset_path=benchmark_dataset_path, max_batch_size=max_batch_size)
        pbar.update(1)

if SQUEEZEBITS_N1_EXP3:
    BUILD_CMD = "trtllm-bench --model meta-llama/Meta-Llama-3-8B build --max_batch_size 256 --max_num_tokens 8192 --dataset {build_dataset_path}"
    BENCHMARK_CMD = "/app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark --engine_dir /tmp/meta-llama/Meta-Llama-3-8B/tp_1_pp_1/ --dataset {benchmark_dataset_path} --request_rate {request_rate} --log_iteration_data --streaming true"
    runner = TensorRTLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD, build_cmd=BUILD_CMD, log_dir=Path("/code/output/TensorRT-LLM"), envs={"CUDA_VISIBLE_DEVICES": "3"})

    generator = TensorRTLLMDatasetGenerator(script_path=Path("/app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py"), save_dir=Path("/code/_TensorRT-LLM/datasets"))
    REQUEST_RATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pbar = tqdm(total = len(REQUEST_RATES))
    is_builded = False
    for request_rate in REQUEST_RATES:
        filename = "2048x128_sample_512"
        build_dataset_path = generator.generate(filename, "meta-llama/Meta-Llama-3-8B", input_len=2048, output_len=128, num_requests=512, for_build=True)
        benchmark_dataset_path = generator.generate(filename, "meta-llama/Meta-Llama-3-8B", input_len=2048, output_len=128, num_requests=512)
        runner.set_log_prefix("n1_exp3")
        runner.init(build_dataset_path=build_dataset_path, benchmark_dataset_path=benchmark_dataset_path, request_rate=request_rate)

        if is_builded is False:
            runner.build_model()
            is_builded = True

        runner.run_benchmark()

        pbar.update(1)

if SQUEEZEBITS_N2_EXP1:
    MAX_BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256, 512]
    MAX_NUM_OF_TOKENS = [1024, 2048, 4096, 8192, 16384]

    BUILD_CMD = "trtllm-bench --model meta-llama/Meta-Llama-3-8B build --max_batch_size {max_batch_size} --max_num_tokens {max_num_tokens} --dataset {build_dataset_path}"
    BENCHMARK_CMD = "/app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark --engine_dir /tmp/meta-llama/Meta-Llama-3-8B/tp_1_pp_1/ --dataset {benchmark_dataset_path} --log_iteration_data --streaming true"
    runner = TensorRTLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD, build_cmd=BUILD_CMD, log_dir=Path("/code/output/TensorRT-LLM"), envs={"CUDA_VISIBLE_DEVICES": "3"})

    generator = TensorRTLLMDatasetGenerator(script_path=Path("/app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py"), save_dir=Path("/code/_TensorRT-LLM/datasets"))

    pbar = tqdm(total=len(MAX_BATCH_SIZES)*len(MAX_NUM_OF_TOKENS))

    for max_batch_size in MAX_BATCH_SIZES:
        for max_num_tokens in MAX_NUM_OF_TOKENS:
            filename = "128x768_sample_1024"
            build_dataset_path = generator.generate(filename, "meta-llama/Meta-Llama-3-8B", input_len=128, output_len=768, num_requests=1024, for_build=True)
            benchmark_dataset_path = generator.generate(filename, "meta-llama/Meta-Llama-3-8B", input_len=128, output_len=768, num_requests=1024)
            runner.set_log_prefix("n2_exp1_decode_heavy")
            runner.run(build_dataset_path=build_dataset_path, benchmark_dataset_path=benchmark_dataset_path, max_batch_size=max_batch_size, max_num_tokens=max_num_tokens)

if SQUEEZEBITS_N2_EXP2:
    MAX_BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256, 512]
    MAX_NUM_OF_TOKENS = [1024, 2048, 4096, 8192, 16384]

    BUILD_CMD = "trtllm-bench --model meta-llama/Meta-Llama-3-8B build --max_batch_size {max_batch_size} --max_num_tokens {max_num_tokens} --dataset {build_dataset_path}"
    BENCHMARK_CMD = "/app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark --engine_dir /tmp/meta-llama/Meta-Llama-3-8B/tp_1_pp_1/ --dataset {benchmark_dataset_path} --log_iteration_data --streaming true"
    runner = TensorRTLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD, build_cmd=BUILD_CMD, log_dir=Path("/code/output/TensorRT-LLM"), envs={"CUDA_VISIBLE_DEVICES": "3"})

    generator = TensorRTLLMDatasetGenerator(script_path=Path("/app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py"), save_dir=Path("/code/_TensorRT-LLM/datasets"))

    pbar = tqdm(total=len(MAX_BATCH_SIZES)*len(MAX_NUM_OF_TOKENS))

    for max_batch_size in MAX_BATCH_SIZES:
        for max_num_tokens in MAX_NUM_OF_TOKENS:
            filename = "768x128_sample_1024"
            build_dataset_path = generator.generate(filename, "meta-llama/Meta-Llama-3-8B", input_len=768, output_len=128, num_requests=1024, for_build=True)
            benchmark_dataset_path = generator.generate(filename, "meta-llama/Meta-Llama-3-8B", input_len=768, output_len=128, num_requests=1024)
            runner.set_log_prefix("n2_exp1_prefill_heavy")
            runner.run(build_dataset_path=build_dataset_path, benchmark_dataset_path=benchmark_dataset_path, max_batch_size=max_batch_size, max_num_tokens=max_num_tokens)

if SQUEEZEBITS_N5_EXP1:
    SUBSETS = ['1k', '2k', '4k', '8k']

    BUILD_CMD = "trtllm-bench --model meta-llama/Meta-Llama-3-8B build --max_batch_size 256 --max_num_tokens 16384 --dataset {build_dataset_path}"
    BENCHMARK_CMD = "/app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark --engine_dir /tmp/meta-llama/Meta-Llama-3-8B/tp_1_pp_1/ --dataset {benchmark_dataset_path} --eos_id 13 --log_iteration_data --streaming true"
    runner = TensorRTLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD, build_cmd=BUILD_CMD, log_dir=Path("/code/output/TensorRT-LLM"), envs={"CUDA_VISIBLE_DEVICES": "3"})

    generator = TensorRTLLMDatasetGenerator(script_path=Path("/app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py"), save_dir=Path("/code/_TensorRT-LLM/datasets"))

    pbar = tqdm(total=len(SUBSETS))

    for subset in SUBSETS:
        build_dataset_path, benchmark_dataset_path = generator.get_dynamic_sonnet_dataset(subset)

        runner.set_log_prefix("n5_exp1_dynamic")
        runner.run(build_dataset_path=build_dataset_path, benchmark_dataset_path=benchmark_dataset_path, subset=subset)
        pbar.update(1)

if SQUEEZEBITS_N5_EXP2:
    SUBSETS = ['1k', '2k', '4k', '8k']
    dynamic_dataset = {
        "1k": {
            "input_len": 512,
            "output_len": 1024,
        },
        "2k": {
            "input_len": 1017,
            "output_len": 1024,
        },
        "4k": {
            "input_len": 3076,
            "output_len": 1024,
        },
        "8k": {
            "input_len": 7154,
            "output_len": 1024,
        }
    }

    BUILD_CMD = "trtllm-bench --model meta-llama/Meta-Llama-3-8B build --max_batch_size 256 --max_num_tokens 16384 --dataset {build_dataset_path}"
    BENCHMARK_CMD = "/app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark --engine_dir /tmp/meta-llama/Meta-Llama-3-8B/tp_1_pp_1/ --dataset {benchmark_dataset_path} --log_iteration_data --streaming true"
    runner = TensorRTLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD, build_cmd=BUILD_CMD, log_dir=Path("/code/output/TensorRT-LLM"), envs={"CUDA_VISIBLE_DEVICES": "3"})

    generator = TensorRTLLMDatasetGenerator(script_path=Path("/app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py"), save_dir=Path("/code/_TensorRT-LLM/datasets"))

    pbar = tqdm(total=len(SUBSETS))

    for subset in SUBSETS:
        dataset_config = dynamic_dataset[subset]
        input_len = dataset_config['input_len']
        output_len = dataset_config['output_len']

        filename = f'fixed_dataset_{input_len}x{output_len}'
        build_dataset_path = generator.generate(filename, "meta-llama/Meta-Llama-3-8B", input_len=input_len, output_len=output_len, num_requests=1024, for_build=True)
        benchmark_dataset_path = generator.generate(filename, "meta-llama/Meta-Llama-3-8B", input_len=input_len, output_len=output_len, num_requests=1024)

        runner.set_log_prefix("n5_exp1_fixed")
        runner.run(build_dataset_path=build_dataset_path, benchmark_dataset_path=benchmark_dataset_path, subset=subset)
        pbar.update(1)