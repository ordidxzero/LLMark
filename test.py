from llmark.utils import CommandTemplateV2
import unittest
from pathlib import Path

class CommandTemplateTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def setUp(self) -> None:
        def dict_to_str(d):
            s = []
            for k, v in d.items():
                s.append(f"{k}_{v}")
            
            return "_".join(s) + '.log'

        self._template = CommandTemplateV2(template="vllm serve {model} --test {test} --vv {vv}")
        self._example_args = {"model": "2", "test": "2", "vv": "1"}
        self._default_log_path = Path(".") / dict_to_str(self._example_args)

    def default_test(self):
        self._template.format(**self._example_args)
        self.assertEqual(self._template.as_string(), f"vllm serve 2 --test 2 --vv 1 >> {self._default_log_path.absolute().as_posix()}")

    def env_test(self):
        self._template.set_envs(TEST="123", VVVV='ASDFSDF')
        self._template.format(**self._example_args)
        self.assertEqual(self._template.as_string(), f"TEST=123 VVVV=ASDFSDF vllm serve 2 --test 2 --vv 1 >> {self._default_log_path.absolute().as_posix()}")

    def log_test(self):
        self._template.set_log(dir=Path("./output/vLLM"), prefix="testtest", use_stdout=False)
        self._template.format(**self._example_args)
        self.assertEqual(self._template.as_string(), f"vllm serve 2 --test 2 --vv 1 >> {(Path('./output/vLLM') / self._default_log_path.name).absolute().as_posix()}")
  
    def log_tes2t(self):
        self._template.set_log(dir=Path("./output/vLLM"), prefix="testtest", use_stdout=True)
        self._template.format(**self._example_args)
        self.assertEqual(self._template.as_string(), f"vllm serve 2 --test 2 --vv 1")