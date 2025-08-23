import subprocess

def get_available_gpu_devices(return_single=False, raise_error=True, gpu_threshold=5, memory_threshold=5):
    cmd = 'nvidia-smi --query-gpu index,utilization.gpu,memory.used,memory.total --format csv,nounits,noheader'
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip().split(", ") for line in lines if line.strip() != '' ]

    available_devices = []

    for line in lines:
        index, gpu_utilization, used_memory, total_memory = list(map(int, line))
        memory_utilization = round((used_memory / total_memory) * 100, 2)

        if gpu_utilization < gpu_threshold and memory_utilization < memory_threshold:
            available_devices.append(str(index))

    if len(available_devices) == 0:
        if raise_error:
            raise Exception("Error: No GPU device available.")
        else:
            return None

    if return_single:
        return available_devices[0]
    else:
        return ','.join(available_devices)

if __name__ == '__main__':
    print(get_available_gpu_devices())