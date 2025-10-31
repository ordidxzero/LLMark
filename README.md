# LLMark
vLLM 및 TensorRT-LLM 벤치마킹을 쉽게 자동화할 수 있는 라이브러리입니다.
## Features
- [x] vLLM 벤치마킹 자동화
- [ ] TensorRT-LLM 벤치마킹 자동화
- [ ] Triton Inference Server 벤치마킹 자동화

## Prerequisites
OS: Linux
Python: 3.8 - 3.12

## Installation and Quickstart
1. LLMark 클론 / 가상 환경 생성 / `typing_extensions` 설치
```bash
git clone https://github.com/ordidxzero/LLMark
cd LLMark
uv venv --python 3.10 --seed
source .venv/bin/activate
uv pip install typing_extensions
```
2. vLLM 설치
```bash
uv pip install vllm
```
3. vLLM 클론 (Benchmark 코드를 얻기 위함)
```bash
mkdir _vllm
cd _vllm
git clone https://github.com/vllm-project/vllm.git .
cd ..
```
4. 코드 작성
```python
from llmark.vllm import VLLMBenchmarkRunner
from pathlib import Path
from tqdm import tqdm

BENCHMARK_CMD = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 1024"
SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs {max_num_seqs}"
runner = VLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD, log_dir=Path("./output"))

INPUT_LENGTHS = [2048, 128]
OUTPUT_LENGTHS = [2048, 128]
MAX_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

pbar = tqdm(total= len(OUTPUT_LENGTHS) * len(INPUT_LENGTHS) * len(MAX_BATCH_SIZES))

for output_len in OUTPUT_LENGTHS:
    for input_len in INPUT_LENGTHS:
        for max_num_seqs in MAX_BATCH_SIZES:
            runner.set_log_prefix("A100-80GB_exp")
            runner.run(max_num_seqs=max_num_seqs, input_len=input_len, output_len=output_len)
            pbar.update(1)
```

## User Guide
이 라이브러리는 [LangChain의 PromptTemplate](https://python.langchain.com/docs/concepts/prompt_templates/)에서 영감을 얻었습니다.
1. 먼저 f-string 형태의 Skeleton 명령어를 만들어주세요.
```python
BENCHMARK_CMD = "uv run _vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-8B --dataset-name random --random-input-len {input_len} --ignore-eos --random-output-len {output_len} --num-prompts 1024"
SERVER_CMD = "vllm serve meta-llama/Meta-Llama-3-8B --dtype bfloat16 --disable-log-requests --max-num-seqs {max_num_seqs}"
```
`BENCHMARK_CMD`에는 `input_len`과 `output_len` 변수가, `SERVER_CMD`에는 `max_num_seqs` 변수가 들어있습니다. `VLLMBenchmarkRunner`가 이 변수에 값을 주입해서 명령어를 완성합니다.

2. Skeleton 명령어를 `VLLMBenchmarkRunner`에 전달하여 `runner` 인스턴스를 만들어주세요.
```python
runner = VLLMBenchmarkRunner(benchmark_cmd=BENCHMARK_CMD, server_cmd=SERVER_CMD)
```
그 외에 아래 파라미터가 있습니다.
- `log_dir`: 로그가 저장될 폴더 경로. 지정하지 않으면 현재 스크립트의 위치로 설정됩니다.
- `log_prefix`: 로그 파일이름에 붙을 Prefix. 지정하지 않으면 prefix가 붙지 않습니다.
- `envs`: 명령어 앞에 붙일 환경변수. 딕셔너리 형태 (i.e. `{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1,2"}`)로 전달하면 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 vllm serve ...`형태로 명령어 앞에 붙습니다.

3. `runner.run`을 실행합니다.
아래처럼 변수를 전달하면 명령어에 값이 주입됩니다. 명령어에 있는 모든 변수에 값이 주입되어야 합니다.
```python
runner.run(max_num_seqs=256, input_len=2048, output_len=128)
```