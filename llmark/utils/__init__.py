import importlib.util
import os, socket, subprocess
from importlib.metadata import version
from typing import Optional, Dict, Literal, TypedDict, List
from typing_extensions import NotRequired
from pathlib import Path
from string import Formatter

# 패키지 버전 정보 확인 방법: https://stackoverflow.com/questions/20180543/how-do-i-check-the-versions-of-python-modules
def find_package_version(pkg: str):
    return version(pkg)

# 패키지 설치 여부 확인 방법: https://stackoverflow.com/questions/1051254/check-if-python-package-is-installed
def find_package(pkg: str, ver: Optional[str] = None, raise_error: bool = True):
    """패키지가 설치되었는지, 버전이 일치하는지 확인하는 함수"""
    is_installed = importlib.util.find_spec(pkg)

    if not is_installed:
        if raise_error:
            raise ImportError(f"{pkg} is not installed")
        else:
            return False
    
    if ver is None:
        return True
    
    is_version_matched = find_package_version(pkg) == ver

    if not is_version_matched:
        if raise_error:
            raise ImportError(f"{pkg} version is not matched")
        else:
            return False
    
    return True

class CommandTemplate:
    _is_valid: bool
    _skeleton: str
    _log_dir: Path
    _log_prefix: str
    _cmd: str
    _envs: str
    _log_filename: List[str]
    _arg_names: List[str]
    _stdout_log: bool
    def __init__(self, template: str, log_prefix: str = '', stdout_log: bool = False):
        self._skeleton = template
        self._log_dir = Path(".")
        self._cmd = ""
        self._envs = ''
        self._log_filename = []
        self._arg_names = self._get_vars_from_f_string()
        self._log_prefix = f"{log_prefix}_" if log_prefix != "" else ""
        self._is_valid = len(self._arg_names) == 0
        self._stdout_log = stdout_log

    def hydrate(self, **kwargs: str | int | float):
        self._is_valid = True
        self._log_filename = [self._log_prefix] if self._log_prefix != '' else []
        # hydrate
        args: Dict[str, str | int | float] = {}

        for key, value in kwargs.items():
            self._log_filename.append(f"{key}_{value}")
            if key in self._arg_names:
                args[key] = value
        
        self._log_filename.append('.log')

        self._cmd = self._skeleton.format(**args)


    def set_log_dir(self, log_dir: Path) -> None:
        self._log_dir = log_dir

    def set_env(self, envs: Dict[str, str]) -> None:
        self._envs = ''
        for key, value in envs.items():
            self._envs += f'{key}={value} '

    def exec(self) -> None:
        assert self._is_valid == True, "You should invoke `hydrate` method"

        cmd = self._transform()
        os.system(cmd)

    def split(self, sep: str = " ") -> List[str]:
        cmd = self._transform()
        return cmd.split(sep)

    def get_absolute_log_path(self):
        assert len(self._log_filename) != 0, 'You should invoke `hydrate` method'
        log_filename = "_".join(self._log_filename)
        p = self._log_dir / log_filename

        if not self._log_dir.exists():
            self._log_dir.mkdir(parents=True)

        return p.absolute().as_posix()

    def _transform(self):
        cmd = self._envs + self._cmd

        if len(self._log_filename) != 0:
            log_filename = "_".join(self._log_filename)
            log_path = self._log_dir / log_filename
            log_path = log_path.absolute().as_posix()

            if not self._log_dir.exists():
                self._log_dir.mkdir(parents=True)
            
            if not self._stdout_log:
                cmd += " >> " + log_path

        return cmd
    
    def _get_vars_from_f_string(self):
        """f-string에서 변수 이름을 추출하는 메서드"""
        return [fn for _, fn, _, _ in Formatter().parse(self._skeleton) if fn is not None]

RunnerType = Literal["vllm", "tensorrt_llm", "triton", "vllm_rbln"]
ENV = Dict[str, str]

class BenchmarkArgs(TypedDict):
    log_dir: NotRequired[Path]
    envs: NotRequired[ENV]
    
class Benchmark:
    runner_type: Optional[RunnerType]
    _env: Dict[str, str]
    _log_dir: Path
    _cmd: Dict[str, CommandTemplate]

    def __init__(self, log_dir: Optional[Path] = Path("."), envs: ENV | None = None):
        assert log_dir is not None, "Error"
        # BenchmarkRunner Type
        self.runner_type = None

        # Environment Variables 관리
        self._env = os.environ.copy()
        self._env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if envs is not None:
            assert isinstance(envs, dict), "env must be dictionary"
            self._env.update(envs)

        # Path 관리
        self._log_dir = log_dir

        # Command 관리
        self._skeleton_cmd = {}
        self._cmd = {}
    
    def _set_runner_type(self, runner_type: RunnerType):
        """Runner Type을 지정하는 메서드. __init__ 메서드에서 반드시 호출되어야 한다"""
        assert runner_type in ["vllm", "tensorrt_llm", "triton", "rbln"], "Invalid RunnerType"
        self.runner_type = runner_type

    @staticmethod
    def is_port_open(host:str  = "localhost", port: int = 8000) -> bool:
        """포트가 열려 있는지 확인하는 메서드"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result == 0
    
    @staticmethod
    def is_docker() -> bool:
        """Docker 환경인지 체크하는 메서드"""
        cgroup = Path('/proc/self/cgroup')
        return Path('/.dockerenv').is_file() or (cgroup.is_file() and 'docker' in cgroup.read_text())

    @staticmethod
    def is_rbln() -> bool:
        """Rebellions NPU 환경을 지원하는지 체크하는 메서드"""
        try:
            subprocess.check_output("rbln-stat")
            return True
        except Exception:
            return False
        
