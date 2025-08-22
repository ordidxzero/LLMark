import importlib.util
import os, socket, subprocess, torch
from importlib.metadata import version
from dataclasses import dataclass, field
from typing import Optional, Dict, Literal, TypedDict, List
from typing_extensions import NotRequired
from pathlib import Path
from string import Formatter, Template
from tqdm import tqdm

RunnerType = Literal["vllm", "tensorrt_llm", "triton", "vllm_rbln"]
ENV = Dict[str, str]

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

@dataclass
class LogState:
    dir: Path = Path(".")
    prefix: str = ''
    device_prefix: str = ''
    use_stdout: bool = False
    filename: Optional[str] = None
    valid_args: Dict[str, str | int | float] = field(default_factory=dict)

    def set_filename(self):
        filename = []

        for key, value in self.valid_args.items():
            filename.append(f"{key}_{value}")

        filename = "_".join(filename)
        filename += '.log'

        if self.prefix != '':
            self.filename = f"{self.device_prefix}_{self.prefix}_{filename}"
        else:
            self.filename = f"{self.device_prefix}_{filename}"
    
    def get_path(self) -> Path:
        assert self.filename is not None, "Error"
        return self.dir / self.filename
    
    def get_absolute_path(self) -> str:
        assert self.filename is not None, "Error"
        return (self.dir / self.filename).absolute().as_posix()
    
    def __post_init__(self):
        device_name = torch.cuda.get_device_name(0)
        device_name = device_name.replace("NVIDIA ", "")
        device_name = device_name.replace(" ", "_")
        self.device_prefix = device_name

class CommandTemplateV2:
    _skeleton: Template
    _envs: Dict[str, str]
    _log_state: LogState
    _arg_names: List[str]
    _cmd: Optional[str]
    def __init__(self, template: str, partial_variables: Dict[str, str | int | float] = {}):
        self._skeleton = Template(template)
        self._skeleton = Template(self._skeleton.safe_substitute(partial_variables))
        self._envs = {}
        self._log_state = LogState(valid_args=partial_variables)
        self._arg_names = self._extract_template_vars()
        self._cmd = None

    def format(self, **kwargs: str | int | float):
        """
        필요한 변수들이 완전히 주어졌을 때, 템플릿을 기반으로 명령어 문자열을 생성합니다.

        Args:
            **kwrags (str | int | float): 템플릿에 삽입될 변수들
        """
        args: Dict[str, str | int | float] = {}

        for key, value in kwargs.items():
            if not isinstance(value, str) or "/" not in value:
                self._log_state.valid_args[key] = value
            if key in self._arg_names:
                args[key] = value

        self._cmd = self._skeleton.template.format(**args)

    def set_envs(self, **envs: str):
        """
        실행 시 사용할 환경 변수들을 설정합니다.

        Args:
            **envs (str): 키워드 인수 형식의 환경 변수 설정 (예: FOO="bar", DEBUG="1")
        """
        self._envs.update(envs)
        return
    
    def set_log(self, **kwargs: Path | str | bool):
        """
        로그 파일을 저장할 디렉토리와 파일명 접두어를 설정합니다.

        Args:
            dir (Path): 로그 파일을 저장할 디렉토리 경로
            prefix (str, optional): 로그 파일 이름에 사용할 접두어. 기본값은 빈 문자열입니다.
            use_stdout (bool, optional): stdout으로 출력할지 여부를 결정합니다.
        """
        for key, value in kwargs.items():
            if key in ['dir', 'prefix', 'use_stdout']:
                if key == 'dir':
                    self._log_state.dir = value
                elif key == 'prefix':
                    self._log_state.prefix = value
                else:
                    self._log_state.use_stdout = value
    
    def _extract_template_vars(self) -> List[str]:
        """
        템플릿 문자열로부터 `{변수}` 형식의 변수명을 추출합니다.

        Returns:
            List[str]: 추출된 변수명들의 리스트
        """
        return [fn for _, fn, _, _ in Formatter().parse(self._skeleton.template) if fn is not None]

    def run(self):
        """
        설정된 환경 및 템플릿을 기반으로 명령어를 실행합니다.

        Raises:
            RuntimeError: 템플릿이 아직 포맷되지 않은 경우 (format() 호출 누락)

        Returns:
            int: 명령어 실행 후의 종료 코드 (0이면 성공)
        """
        cmd = self._transform()
        os.system(cmd)

    def as_string(self) -> str:
        """
        현재 상태에서 구성된 최종 명령어 문자열을 반환합니다.

        Returns:
            str: 실행 가능한 명령어 문자열
        """ 
        return self._transform()

    def _transform(self) -> str:
        """
        환경 변수 및 로그 리디렉션 설정을 포함하여 전체 명령어 문자열을 구성합니다.

        Returns:
            str: 로그 설정과 환경변수가 포함된 실행 가능한 최종 명령어 문자열
        """
        assert self._cmd is not None

        envs = ''
        if len(self._envs) != 0:
            for key, value in self._envs.items():
                envs += f'{key.upper()}={value} '
        
        self._log_state.set_filename()
        log_path = self._log_state.get_absolute_path()

        if not self._log_state.dir.exists():
            self._log_state.dir.mkdir(parents=True)

        cmd = envs + self._cmd

        if not self._log_state.use_stdout:
            cmd += f" >> {log_path}"

        return cmd

    @property
    def log_dir(self) -> Path:
        return self._log_state.dir
    
    @property
    def log_path(self) -> Path:
        return self._log_state.get_path()

class BenchmarkV2:
    runner_type: Optional[RunnerType]
    _env: Dict[str, str]
    _user_env: Dict[str, str]
    _log_dir: Path
    _cmd: Dict[str, CommandTemplateV2]
    _is_ready: bool

    def __init__(self, log_dir: Optional[Path] = Path("."), envs: ENV | None = None):
        assert log_dir is not None, "Error"
        # BenchmarkRunner Type
        self.runner_type = None

        # Environment Variables 관리
        self._env = os.environ.copy()
        self._user_env = {}
        self._env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        self._user_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if envs is not None:
            assert isinstance(envs, dict), "env must be dictionary"
            self._env.update(envs)
            self._user_env.update(envs)

        # Path 관리
        self._log_dir = log_dir
        self._is_ready = False

        # Command 관리
        self._cmd = {}

    def init(self, **kwargs: str | int | float):
        for _, cmd in self._cmd.items():
            cmd.format(**kwargs)

        self._is_ready = True

    def set_prefix(self, prefix: str):
        for _, cmd in self._cmd.items():
            cmd.set_log(prefix=prefix)
    
    def _set_runner_type(self, runner_type: RunnerType):
        """Runner Type을 지정하는 메서드. __init__ 메서드에서 반드시 호출되어야 한다"""
        assert runner_type in ["vllm", "tensorrt_llm", "triton", "vllm_rbln"], "Invalid RunnerType"
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

class LogAnalyzerV2:
    pass

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
    def __init__(self, template: str, stdout_log: bool = False):
        self._skeleton = template
        self._log_dir = Path(".")
        self._cmd = ""
        self._envs = ''
        self._log_filename = []
        self._arg_names = self._get_vars_from_f_string()
        self._log_prefix = ""
        self._is_valid = len(self._arg_names) == 0
        self._stdout_log = stdout_log

    def hydrate(self, **kwargs: str | int | float):
        self._is_valid = True
        self._log_filename = [self._log_prefix] if self._log_prefix != '' else []
        # hydrate
        args: Dict[str, str | int | float] = {}

        for key, value in kwargs.items():
            if not isinstance(value, str) or "/" not in value:
                self._log_filename.append(f"{key}_{value}")
            if key in self._arg_names:
                args[key] = value
        
        self._log_filename.append('.log')

        self._cmd = self._skeleton.format(**args)


    def set_log_dir(self, log_dir: Path) -> None:
        self._log_dir = log_dir

    @property
    def log_dir(self):
        return self._log_dir

    def set_env(self, envs: Dict[str, str]) -> None:
        self._envs = ''
        for key, value in envs.items():
            self._envs += f'{key.upper()}={value} '
    
    def set_log_prefix(self, prefix: str) -> None:
        self._log_prefix = prefix

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

class BenchmarkArgs(TypedDict):
    log_dir: NotRequired[Path]
    envs: NotRequired[ENV]
    
class Benchmark:
    runner_type: Optional[RunnerType]
    _env: Dict[str, str]
    _user_env: Dict[str, str]
    _log_dir: Path
    _cmd: Dict[str, CommandTemplate]

    def __init__(self, log_dir: Optional[Path] = Path("."), envs: ENV | None = None):
        assert log_dir is not None, "Error"
        # BenchmarkRunner Type
        self.runner_type = None

        # Environment Variables 관리
        self._env = os.environ.copy()
        self._user_env = {}
        self._env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        self._user_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if envs is not None:
            assert isinstance(envs, dict), "env must be dictionary"
            self._env.update(envs)
            self._user_env.update(envs)

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
        
class LogAnalyzer:
    def __init__(self, log_dir: Path):
        self._log_dir = log_dir

class LogParser:
    def __init__(self, log_dir: Path):
        self._log_dir = log_dir
    
    def run(self, pattern: str, output_dir: Path = Path("."), disable_tqdm: bool = False):
        assert pattern is not None and pattern != "" and isinstance(pattern, str), "Logfile Pattern is required"

        logfile_iterator = self._log_dir.glob(pattern)

        if not disable_tqdm:
            logfile_iterator = tqdm(logfile_iterator)

        for logfile_path in logfile_iterator:
            pass