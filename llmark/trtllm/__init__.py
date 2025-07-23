from llmark.utils import Benchmark, BenchmarkArgs, CommandTemplate
from typing_extensions import Unpack
from datasets import load_dataset
from pathlib import Path
import os, torch, json

class TensorRTLLMDatasetGenerator:
    _dataset_dir: Path
    _script_path: Path
    def __init__(self, script_path: Path, save_dir: Path = Path(".")):
        self._dataset_dir = save_dir
        self._script_path = script_path

    def generate(self, name: str, tokenizer: str = "meta-llama/Llama-3.1-8B-Instruct", num_requests: int = 1024, input_len: int = 1024, input_stdev: int = 0, output_len: int = 1024, output_stdev: int = 0, for_build: bool = False) -> str:
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


        cmd = f"python {self._script_path} --output {dataset_path} --tokenizer {tokenizer} token-norm-dist --num-requests {num_requests} --input-mean {input_len} --input-stdev {input_stdev} --output-mean {output_len} --output-stdev {output_stdev}"
        
        if for_build:
            cmd = f"python {self._script_path} --stdout --tokenizer {tokenizer} token-norm-dist --input-mean {input_len} --output-mean {output_len} --input-stdev {input_stdev} --output-stdev {output_stdev} --num-requests {num_requests} > {dataset_path}"
        
        os.system(cmd)

        return dataset_path
    
    def _get_dynamic_sonnet_metadata(self, config):
        input_mean = config['input_len']
        input_stdev = config['input_stdev']
        output_mean = config['output_len']
        num_requests = config['num_requests']
        max_input_len = config['max_input_len']
        return {"workload_type": "token-norm-dist", "input_mean": input_mean, "input_stdev": input_stdev, "output_mean": output_mean, "output_stdev": 0, "num_requests": num_requests, "tokenize_vocabsize": 128000, "max_input_len": max_input_len, "max_output_len": output_mean, "workload_name": f"workload_type:token-norm-dist__input_mean:{input_mean}__input_stdev:{input_stdev}__output_mean:{output_mean}__output_stdev:0__num_requests:{num_requests}__tokenize_vocabsize:128000__max_input_len:{max_input_len}__max_output_len:{output_mean}"}

    def _to_sample(self, data, output_len):
        tok_inputs = data['tok_inputs']
        return {"input_len": len(tok_inputs),"input_ids":tok_inputs,"output_len": output_len, "task_id": -1}

    def get_dynamic_sonnet_dataset(self, hf_split: str = '1k', output_len: int = 1024):
        build_dataset_path = self._dataset_dir / f"dynamic_sonnet_{hf_split}.txt"
        benchmark_dataset_path = self._dataset_dir / f'dynamic_sonnet_{hf_split}.json'

        if build_dataset_path.exists() and benchmark_dataset_path.exists():
            return build_dataset_path.absolute().as_posix(), benchmark_dataset_path.absolute().as_posix()

        dataset = load_dataset('squeezebits/dynamic_sonnet_llama3', split=hf_split, streaming=True)
        dataset = dataset.shuffle()

        dynamic_dataset = {
            "1k": {
                "input_len": 512,
                "input_stdev": 140,
                "output_len": 1024,
                "num_requests": 1024,
                "max_input_len": 773,
            },
            "2k": {
                "input_len": 1017,
                "input_stdev": 288,
                "output_len": 1024,
                "num_requests": 1024,
                "max_input_len": 1536,
            },
            "4k": {
                "input_len": 3076,
                "input_stdev": 294,
                "output_len": 1024,
                "num_requests": 1024,
                "max_input_len": 3601,
            },
            "8k": {
                "input_len": 7154,
                "input_stdev": 284,
                "output_len": 1024,
                "num_requests": 1024,
                "max_input_len": 7709,
            }
        }

        metadata = self._get_dynamic_sonnet_metadata(dynamic_dataset[hf_split])
        samples = []

        with open(build_dataset_path) as f:
            for data in dataset:
                tok_inputs = data['tok_inputs']
                task_id = data['id']

                line = {"task_id":task_id,"logits":tok_inputs,"output_tokens":output_len}

                samples.append(self._to_sample(data, output_len))

                f.write(json.dumps(line))
                f.write("\n")

        
        with open(benchmark_dataset_path) as f:
            d = {"metadata": metadata, "samples": samples}
            json.dump(d, f)

        return build_dataset_path.absolute().as_posix(), benchmark_dataset_path.absolute().as_posix()


class TensorRTLLMBenchmarkRunner(Benchmark):
    def __init__(self, benchmark_cmd: str, build_cmd: str, delete_cmd: str = "rm -rf /tmp/meta-llama", **kwargs: Unpack[BenchmarkArgs]):
        super().__init__(**kwargs)

        self._set_runner_type("tensorrt_llm")

        self._cmd["benchmark"] = CommandTemplate(benchmark_cmd)
        self._cmd["build"] = CommandTemplate(build_cmd, stdout_log=True)
        self._cmd["delete"] = CommandTemplate(delete_cmd, stdout_log=True)

    def init(self, **kwargs: str | int | float):
        self._cmd['benchmark'].set_log_dir(self._log_dir)

        for _, cmd in self._cmd.items():
            cmd.hydrate(**kwargs)

    def run(self, **kwargs: str | int | float):
        self.init(**kwargs)

        self.build_model()
        self.run_benchmark()

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
        self._cmd['delete'].exec()
        self._cmd["build"].set_env(self._user_env)
        self._cmd["build"].exec()

    def run_benchmark(self):
        print("Start Benchmark...")
        
        self._cmd["benchmark"].set_env(self._user_env)
        self._cmd["benchmark"].exec()


class TritonBenchmarkRunner(Benchmark):
    def __init__(self, benchmark_cmd: str, build_cmd: str):
        pass