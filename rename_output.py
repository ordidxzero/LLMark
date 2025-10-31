from pathlib import Path
import os

output_path = Path("./output/vLLM")

dir_list = os.listdir(output_path)

for d in dir_list:
    d_path = output_path / d

    for l in d_path.glob(f"**/*quantization_gptq*.log"):
        a = l.with_name(l.name.replace("quantization_gptq_marlin_", ""))
        l.rename(a)