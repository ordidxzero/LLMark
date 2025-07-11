from llmark.utils import Benchmark, BenchmarkArgs, CommandTemplate
from typing_extensions import Unpack
from pathlib import Path
import os

class TensorRTLLMDatasetGenerator:
    _dataset_dir: Path
    _script_path: Path
    def __init__(self, script_path: Path, save_dir: Path = Path(".")):
        self._dataset_dir = save_dir
        self._script_path = script_path

    def generate(self, name: str, tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct", num_requests: int = 1024, input_len: int = 1024, output_len: int = 1024) -> str:
        if not name.endswith(".json"):
            name += ".json"

        dataset_path = (self._dataset_dir / name).absolute().as_posix()

        cmd = f"python {self._script_path} --output {dataset_path} --tokenizer {tokenizer} token-norm-dist --num-requests {num_requests} --input-mean {input_len} --input-stdev 0 --output-mean {output_len} --output-stdev 0"
        os.system(cmd)

        return dataset_path

class TensorRTLLMBenchmarkRunner(Benchmark):
    def __init__(self, benchmark_cmd: str, build_cmd: str, delete_cmd: str = "rm -rf /tmp/meta-llama", **kwargs: Unpack[BenchmarkArgs]):
        super().__init__(**kwargs)

        self._set_runner_type("tensorrt_llm")

        self._cmd["benchmark"] = CommandTemplate(benchmark_cmd)
        self._cmd["build"] = CommandTemplate(build_cmd)
        self._cmd["delete"] = CommandTemplate(delete_cmd)

    def run(self, **kwargs: str | int | float):
        self._cmd['benchmark'].set_log_dir(self._log_dir)

        for _, cmd in self._cmd.items():
            cmd.hydrate(**kwargs)

        self._build_model()
        self._run_benchmark()

        self._cmd['delete'].exec()


    def _build_model(self):
        self._cmd["build"].set_env(self._env)
        self._cmd["build"].exec()

    def _run_benchmark(self):
        print("Start Benchmark...")
        
        self._cmd["benchmark"].set_env(self._env)
        self._cmd["benchmark"].exec()


class TritonBenchmarkRunner(Benchmark):
    def __init__(self, benchmark_cmd: str, build_cmd: str):
        pass