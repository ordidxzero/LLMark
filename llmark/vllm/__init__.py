import subprocess, time, torch, gc
from typing_extensions import Unpack, Callable
from pathlib import Path
from llmark.utils import (
    find_package,
    find_package_version,
    Benchmark,
    BenchmarkV2,
    BenchmarkArgs,
    CommandTemplate,
    CommandTemplateV2,
    CommandArgs
)

class VLLMBenchmarkRunnerV2(BenchmarkV2):
    _server_process: subprocess.Popen[str] | None
    _terminate_server: Callable[..., None] | None

    def __init__(
        self, benchmark_cmd: CommandTemplateV2, server_cmd: CommandTemplateV2, enable_nvtx: bool = False, **kwargs: Unpack[BenchmarkArgs]
    ):
        if not find_package("vllm"):
            raise Exception("vLLM is not installed")
        
        
        print("=" * 30)
        print("Current vLLM Version: ", find_package_version("vllm"))
        print("=" * 30)

        super().__init__(**kwargs)

        self._set_runner_type("vllm")

        self._server_process = None
        self._terminate_server = None
        self._enable_nvtx = enable_nvtx

        self._cmd["benchmark"] = benchmark_cmd
        self._cmd["server"] = server_cmd

        self._cmd["benchmark"].set_log(dir=self._log_dir / "benchmark")
        self._cmd["server"].set_log(dir=self._log_dir / "server", use_stdout=True)
        self._cmd["benchmark"].set_envs(**self._user_env)

    def set_benchmark_cmd(self, cmd: CommandTemplateV2):
        self._cmd['benchmark'] = cmd
        self._cmd['benchmark'].set_log(dir=self._log_dir / 'benchmark')
        self._cmd["benchmark"].set_envs(**self._user_env)

    def set_server_cmd(self, cmd: CommandTemplateV2):
        self._cmd['server'] = cmd
        self._cmd["server"].set_log(dir=self._log_dir / "server", use_stdout=True)

    def run_server(self):
        assert self._is_ready == True, "Runner is not initalized. Invoke init()"
        assert Benchmark.is_port_open() == False, "Port 8000 is already used."
        if self._server_process is not None:
            assert (
                self._server_process.poll() == 0
            ), "Server does not closed successfully."
            self._server_process = None

        print("=" * 30)
        print(f"Server Command: {self._cmd['server'].as_string()}")
        print("=" * 30)
        log_file = open(self._cmd["server"].log_path.absolute().as_posix(), "w")

        server_process = subprocess.Popen(
            args=self._cmd["server"].as_string().split(" "),
            text=True,
            encoding="utf-8",
            stdout=log_file,
            env=self._env,
        )
        self._server_process = server_process

        def terminate():
            log_file.close()
            server_process.terminate()
            server_process.wait(timeout=60)
            print("Terminated...")
            torch.cuda.empty_cache()
            gc.collect()

        self._terminate_server = terminate
        log_file.close()

    def run_benchmark(self):
        print("=" * 30)
        print(f"Benchmark Command: {self._cmd['benchmark'].as_string()}")
        print("=" * 30)
        self._cmd["benchmark"].run()

    def run(self, **kwargs: str | int | float):
        """init, run_server, run_benchmark를 합친 메서드"""
        self.init(**kwargs)
        self.run_server()

        assert self._terminate_server != None, "Server is not initialized"

        while True:
            if Benchmark.is_port_open():
                self.run_benchmark()
                break
            time.sleep(1)

        ENABLE_NSYS_PROFILE = self._cmd["server"].as_string().startswith("nsys profile")

        if "VLLM_TORCH_PROFILER_DIR" in self._user_env:
            print("Wait to flush out")
            time.sleep(180)

        self._terminate_server()
        self._is_ready = False

class RBLNVLLMBenchmarkRunner(BenchmarkV2):
    _server_process: subprocess.Popen[str] | None
    _terminate_server: Callable[..., None] | None

    def __init__(
        self, benchmark_cmd: CommandTemplateV2, server_cmd: CommandTemplateV2, **kwargs: Unpack[BenchmarkArgs]
    ):
        if not BenchmarkV2.is_rbln():
            raise Exception("This BenchmarkRunner only support Rebellions NPU.")

        if not find_package("vllm"):
            raise Exception("vLLM is not installed")
        
        print("=" * 30)
        print("Current vLLM Version: ", find_package_version("vllm"))
        print("=" * 30)

        super().__init__(**kwargs)

        self._set_runner_type("vllm_rbln")

        self._server_process = None
        self._terminate_server = None

        self._cmd["benchmark"] = benchmark_cmd
        self._cmd["server"] = server_cmd

        self._cmd["benchmark"].set_log(dir=self._log_dir / "benchmark")
        self._cmd["server"].set_log(dir=self._log_dir / "server", use_stdout=True)
        self._cmd["benchmark"].set_envs(**self._user_env)

class VLLMBenchmarkRunner(Benchmark):
    _server_process: subprocess.Popen[str] | None
    _terminate_server: Callable[..., None] | None
    _is_init: bool

    def __init__(
        self, benchmark_cmd: str, server_cmd: str, **kwargs: Unpack[BenchmarkArgs]
    ):
        super().__init__(**kwargs)

        self._set_runner_type("vllm")

        self._server_process = None
        self._terminate_server = None

        self._cmd["benchmark"] = CommandTemplate(benchmark_cmd)
        self._cmd["server"] = CommandTemplate(server_cmd, stdout_log=True)
        self._is_init = False

    def init(self, **kwargs: str | int | float):
        self._cmd["benchmark"].set_log_dir(self._log_dir / "client")
        self._cmd["server"].set_log_dir(self._log_dir / "server")

        for _, cmd in self._cmd.items():
            cmd.hydrate(**kwargs)

        self._is_init = True

    def set_benchmark_cmd(self, cmd: str):
        self._cmd["benchmark"] = CommandTemplate(cmd)

    def set_server_cmd(self, cmd: str):
        self._cmd["server"] = CommandTemplate(cmd, stdout_log=True)

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

    def get_server_log(self) -> Path:
        server_cmd = self._cmd['server']

        return server_cmd.log_dir

    def run_server(self):
        assert self._is_init == True, "Runner is not initalized. Invoke init()"
        assert Benchmark.is_port_open() == False, "Port 8000 is already used."
        if self._server_process is not None:
            assert (
                self._server_process.poll() == 0
            ), "Server does not closed successfully."
            self._server_process = None

        print("Running...")
        log_file = open(self._cmd["server"].get_absolute_log_path(), "w")

        server_process = subprocess.Popen(
            args=self._cmd["server"].split(" "),
            text=True,
            encoding="utf-8",
            stdout=log_file,
            env=self._env,
        )
        self._server_process = server_process

        def terminate():
            log_file.close()
            server_process.terminate()
            server_process.wait(timeout=60)
            print("Terminated...")
            torch.cuda.empty_cache()
            gc.collect()

        self._terminate_server = terminate
        log_file.close()

    def run_benchmark(self):
        print("Start Benchmark...")
        self._cmd["benchmark"].set_env(self._user_env)
        self._cmd["benchmark"].exec()

    def run(self, **kwargs: str | int | float):
        """init, run_server, run_benchmark를 합친 메서드"""
        self.init(**kwargs)
        self.run_server()

        assert self._terminate_server != None, "Server is not initialized"

        while True:
            if Benchmark.is_port_open():
                self.run_benchmark()
                break
            time.sleep(1)

        if "VLLM_TORCH_PROFILER_DIR" in self._user_env:
            print("Wait to flush out")
            time.sleep(1800)
        self._terminate_server()