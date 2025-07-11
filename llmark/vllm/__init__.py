import subprocess, time
from typing_extensions import Unpack, Callable
from llmark.utils import find_package, find_package_version, Benchmark, BenchmarkArgs, CommandTemplate

class VLLMBenchmarkRunner(Benchmark):
    _server_process: subprocess.Popen[str] | None
    _terminate_server: Callable[..., None] | None
    _is_init: bool
    def __init__(self, benchmark_cmd: str, server_cmd: str, log_prefix: str = '', **kwargs: Unpack[BenchmarkArgs]):
        super().__init__(**kwargs)

        self._set_runner_type("vllm")

        self._server_process = None
        self._terminate_server = None

        self._cmd["benchmark"] = CommandTemplate(benchmark_cmd, log_prefix=log_prefix)
        self._cmd["server"] = CommandTemplate(server_cmd, log_prefix=log_prefix)

        self._is_init = False
    
    def init(self, **kwargs: str | int | float):
        self._cmd["benchmark"].set_log_dir(self._log_dir / "client")
        self._cmd["server"].set_log_dir(self._log_dir / "server")

        for _, cmd in self._cmd.items():
            cmd.hydrate(**kwargs)

        self._is_init = True

    def run_server(self):
        assert self._is_init == True, "Runner is not initalized. Invoke init()"
        assert Benchmark.is_port_open() == False, "Port 8000 is already used."
        if self._server_process is not None:
            assert self._server_process.poll() == 0, "Server does not closed successfully."
            self._server_process = None
        
        print("Running...")
        log_file = open(self._cmd['server'].get_absolute_log_path(), "w")
        
        server_process = subprocess.Popen(args=self._cmd["server"].split(" "), text=True, encoding="utf-8", stdout=log_file, env=self._env)
        self._server_process = server_process

        def terminate():
            log_file.close()
            server_process.terminate()
            server_process.wait(timeout=60)
            print("Terminated...")

        self._terminate_server = terminate

    def run_benchmark(self):
        print("Start Benchmark...")
        self._cmd['benchmark'].exec()

    def run(self, **kwargs: str | int | float):
        """init, run_server, run_benchmark를 합친 메서드"""
        self.init(**kwargs)
        self.run_server()

        assert self._terminate_server != None, "Server is not initialized"

        while True:
            if Benchmark.is_port_open():
                self.run_benchmark()
                self._terminate_server()
                break
            time.sleep(1)

if __name__ != "__main__":
    find_package("vllm")
    print("Current vLLM Version: ", find_package_version('vllm'))