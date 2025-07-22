from llmark.utils import Benchmark, BenchmarkArgs, CommandTemplate
from typing_extensions import Unpack
from pathlib import Path
import os, torch

class TensorRTLLMDatasetGenerator:
    _dataset_dir: Path
    _script_path: Path
    def __init__(self, script_path: Path, save_dir: Path = Path(".")):
        self._dataset_dir = save_dir
        self._script_path = script_path

    def generate(self, name: str, tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct", num_requests: int = 1024, input_len: int = 1024, output_len: int = 1024, for_build: bool = False) -> str:
        if for_build:
            if not name.endswith('.txt'):
                name += '.txt'
        else:
            if not name.endswith(".json"):
                name += ".json"

        if (self._dataset_dir / name).exists():
            print("Skip generating dataset...")
            return (self._dataset_dir / name).absolute().as_posix()

        print("Generate Dataset ...")

        dataset_path = (self._dataset_dir / name).absolute().as_posix()


        cmd = f"python {self._script_path} --output {dataset_path} --tokenizer {tokenizer} token-norm-dist --num-requests {num_requests} --input-mean {input_len} --input-stdev 0 --output-mean {output_len} --output-stdev 0"
        
        if for_build:
            cmd = f"python {self._script_path} --stdout --tokenizer {tokenizer} token-norm-dist --input-mean {input_len} --output-mean {output_len} --input-stdev 0 --output-stdev 0 --num-requests {num_requests} > {dataset_path}"
        
        os.system(cmd)

        return dataset_path

class TensorRTLLMBenchmarkRunner(Benchmark):
    def __init__(self, benchmark_cmd: str, build_cmd: str, delete_cmd: str = "rm -rf /tmp/meta-llama", **kwargs: Unpack[BenchmarkArgs]):
        super().__init__(**kwargs)

        self._set_runner_type("tensorrt_llm")

        self._cmd["benchmark"] = CommandTemplate(benchmark_cmd)
        self._cmd["build"] = CommandTemplate(build_cmd, stdout_log=True)
        self._cmd["delete"] = CommandTemplate(delete_cmd)

    def init(self, **kwargs: str | int | float):
        self._cmd['benchmark'].set_log_dir(self._log_dir)

        for _, cmd in self._cmd.items():
            cmd.hydrate(**kwargs)

    def run(self, **kwargs: str | int | float):
        self.init(**kwargs)

        self.build_model()
        self.run_benchmark()

        self._cmd['delete'].exec()

    def set_log_prefix(self, prefix: str, name: str | None = None):
        device_name = torch.cuda.get_device_name(0)
        device_name = device_name.replace("NVIDIA ", "")
        device_name = device_name.replace(" ", "_")
        prefix = device_name + "_" + prefix
        if name is None:
            for cmd in self._cmd.values():
                cmd.set_log_prefix(prefix)
        else:
            assert name in self._cmd, "Not Found"
            self._cmd[name].set_log_prefix(prefix)


    def build_model(self):
        print("Start Build...")
        self._cmd["build"].set_env(self._user_env)
        self._cmd["build"].exec()

    def run_benchmark(self):
        print("Start Benchmark...")
        
        self._cmd["benchmark"].set_env(self._user_env)
        self._cmd["benchmark"].exec()


class TritonBenchmarkRunner(Benchmark):
    def __init__(self, benchmark_cmd: str, build_cmd: str):
        pass