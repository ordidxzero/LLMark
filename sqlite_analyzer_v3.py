# ================================================
# Graph 모드에서 추출한 nsys-rep를 SQLite로 조작하여
# Decode step 정보를 파악할 수 있게 해주는 툴입니다.
# ================================================


import sqlite3, time, argparse, re
from typing import List, Tuple, Dict
from collections import OrderedDict
from tqdm import tqdm
from llmark.sqlite import get_kernel_mapping, find_boundary_kernel_ids, find_start_kernels, find_prefill_kernel_times
from llmark.sqlite.logger import KernelLogger
from llmark.sqlite.utils import KernelFinder
from llmark.sqlite.float16 import create_virtual_decode_event_for_fp16, create_virtual_prefill_event_for_fp16
from llmark.sqlite.smooth_quant import create_virtual_decode_event_for_smooth_quant

# ================================================
# Analyze DB
# ================================================

def analyze_prefill_fp16(cursor: sqlite3.Cursor):
    PREFILL_QUERY = "SELECT ROUND(AVG((end - start) / 1000000.0), 3) FROM VIRTUAL_NVTX_EVENTS WHERE name = 'PREFILL'"
    cursor.execute(PREFILL_QUERY)
    avg_prefill = cursor.fetchone()[0]

    PREFILL_QUERY = "SELECT start, end FROM VIRTUAL_NVTX_EVENTS WHERE name = 'PREFILL'"
    cursor.execute(PREFILL_QUERY)
    last_prefill = cursor.fetchall()[-1]
    last_prefill_start_time = last_prefill[0]
    last_prefill_end_time = last_prefill[1]

    QKV_PROJ_LATENCY_QUERY = f"SELECT ROUND(AVG((end - start) / 1000.0), 3) FROM VIRTUAL_NVTX_OPS WHERE name='qkv_proj' AND start >= {last_prefill_start_time} AND end <= {last_prefill_end_time}"
    cursor.execute(QKV_PROJ_LATENCY_QUERY)
    qkv_proj_latency = cursor.fetchone()[0]

    SELF_ATTN_LATENCY_QUERY = f"SELECT ROUND(AVG((end - start) / 1000.0), 3) FROM VIRTUAL_NVTX_OPS WHERE name='self_attn' AND start >= {last_prefill_start_time} AND end <= {last_prefill_end_time}"
    cursor.execute(SELF_ATTN_LATENCY_QUERY)
    self_attn_latency = cursor.fetchone()[0]

    O_PROJ_LATENCY_QUERY = f"SELECT ROUND(AVG((end - start) / 1000.0), 3) FROM VIRTUAL_NVTX_OPS WHERE name='o_proj' AND start >= {last_prefill_start_time} AND end <= {last_prefill_end_time}"
    cursor.execute(O_PROJ_LATENCY_QUERY)
    o_proj_latency = cursor.fetchone()[0]

    GATE_UP_PROJ_LATENCY_QUERY = f"SELECT ROUND(AVG((end - start) / 1000.0), 3) FROM VIRTUAL_NVTX_OPS WHERE name='gate_up_proj' AND start >= {last_prefill_start_time} AND end <= {last_prefill_end_time}"
    cursor.execute(GATE_UP_PROJ_LATENCY_QUERY)
    gate_up_proj_latency = cursor.fetchone()[0]

    DOWN_PROJ_LATENCY_QUERY = f"SELECT ROUND(AVG((end - start) / 1000.0), 3) FROM VIRTUAL_NVTX_OPS WHERE name='down_proj' AND start >= {last_prefill_start_time} AND end <= {last_prefill_end_time}"
    cursor.execute(DOWN_PROJ_LATENCY_QUERY)
    down_proj_latency = cursor.fetchone()[0]

    return avg_prefill, qkv_proj_latency, self_attn_latency, o_proj_latency, gate_up_proj_latency, down_proj_latency

def analyze_decode_fp16(cursor: sqlite3.Cursor):
    TPOT_QUERY = "SELECT ROUND(AVG((end - start) / 1000000.0), 3) FROM VIRTUAL_NVTX_EVENTS WHERE name = 'TPOT'"
    cursor.execute(TPOT_QUERY)
    avg_tpot = cursor.fetchone()[0]

    TPOT_QUERY = "SELECT start, end FROM VIRTUAL_NVTX_EVENTS WHERE name = 'TPOT'"
    cursor.execute(TPOT_QUERY)
    last_tpot = cursor.fetchall()[-1]
    last_tpot_start_time = last_tpot[0]
    last_tpot_end_time = last_tpot[1]

    QKV_PROJ_LATENCY_QUERY = f"SELECT ROUND(AVG((end - start) / 1000.0), 3) FROM VIRTUAL_NVTX_OPS WHERE name='qkv_proj' AND start >= {last_tpot_start_time} AND end <= {last_tpot_end_time}"
    cursor.execute(QKV_PROJ_LATENCY_QUERY)
    qkv_proj_latency = cursor.fetchone()[0]

    SELF_ATTN_LATENCY_QUERY = f"SELECT ROUND(AVG((end - start) / 1000.0), 3) FROM VIRTUAL_NVTX_OPS WHERE name='self_attn' AND start >= {last_tpot_start_time} AND end <= {last_tpot_end_time}"
    cursor.execute(SELF_ATTN_LATENCY_QUERY)
    self_attn_latency = cursor.fetchone()[0]

    O_PROJ_LATENCY_QUERY = f"SELECT ROUND(AVG((end - start) / 1000.0), 3) FROM VIRTUAL_NVTX_OPS WHERE name='o_proj' AND start >= {last_tpot_start_time} AND end <= {last_tpot_end_time}"
    cursor.execute(O_PROJ_LATENCY_QUERY)
    o_proj_latency = cursor.fetchone()[0]

    GATE_UP_PROJ_LATENCY_QUERY = f"SELECT ROUND(AVG((end - start) / 1000.0), 3) FROM VIRTUAL_NVTX_OPS WHERE name='gate_up_proj' AND start >= {last_tpot_start_time} AND end <= {last_tpot_end_time}"
    cursor.execute(GATE_UP_PROJ_LATENCY_QUERY)
    gate_up_proj_latency = cursor.fetchone()[0]

    DOWN_PROJ_LATENCY_QUERY = f"SELECT ROUND(AVG((end - start) / 1000.0), 3) FROM VIRTUAL_NVTX_OPS WHERE name='down_proj' AND start >= {last_tpot_start_time} AND end <= {last_tpot_end_time}"
    cursor.execute(DOWN_PROJ_LATENCY_QUERY)
    down_proj_latency = cursor.fetchone()[0]

    return avg_tpot, qkv_proj_latency, self_attn_latency, o_proj_latency, gate_up_proj_latency, down_proj_latency


# ================================================
# ================================================

# ================================================
# Intialization
# ================================================
def connect(filename: str):
    connection = sqlite3.connect(filename)
    cursor = connection.cursor()

    return cursor, connection

def get_max_num_seqs(filename: str):
    nums = re.search(r"max-num-seqs_(\d+)_num-prompts_(\d+)", filename)
    assert nums is not None, "error"
    max_num_seqs = int(nums.group(1))
    num_prompts = int(nums.group(2))
    return max_num_seqs, int(num_prompts / max_num_seqs)

def init(cursor: sqlite3.Cursor, connection: sqlite3.Connection):
    cursor.execute("CREATE TABLE IF NOT EXISTS KERNEL_WITH_INDEX AS SELECT start, end, demangledName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL")
    # cursor.execute("ALTER TABLE KERNEL_WITH_INDEX ADD COLUMN start_bucket INTEGER GENERATED ALWAYS AS (FLOOR(start / 1000000)) VIRTUAL")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_kernel_start ON KERNEL_WITH_INDEX(start)")

    cursor.execute("CREATE TABLE IF NOT EXISTS VIRTUAL_NVTX_OPS (pid INTEGER PRIMARY KEY, name TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL)")
    cursor.execute("CREATE TABLE IF NOT EXISTS VIRTUAL_NVTX_EVENTS (pid INTEGER PRIMARY KEY, name TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL)")
    connection.commit()
    return

# ================================================
# ================================================

def fprint(name: str, value):
    name = name.ljust(15)
    print(name, ":", value)

def main(args: argparse.Namespace):
    if 'smooth_quant' in args.filename:
        args.quantization = 'smooth_quant'

    cursor, connection = connect(args.filename)
    max_num_seqs, num_prefill = get_max_num_seqs(args.filename)
    init(cursor, connection)


    if args.check_prefill_kernel or args.check_decode_kernel:
        logger = KernelLogger(cursor=cursor, max_num_seqs=max_num_seqs)

        if args.check_prefill_kernel:
            logger.print_prefill_kernel()
        
        if args.check_decode_kernel:
            logger.print_decode_kernel()

        return

    finder = KernelFinder(cursor=cursor, max_num_seqs=max_num_seqs, quantization=args.quantization)

    decode_kernel_mapping, prefill_kernel_mapping = get_kernel_mapping(cursor, args.quantization, max_num_seqs)

    prefill_start_kernel_id, prefill_end_kernel_id = finder.get_prefill_boundary_kernel_ids()
    decode_start_kernel_id, decode_end_kernel_id = finder.get_decode_boundary_kernel_ids()

    create_virtual_decode_event = create_virtual_decode_event_for_fp16
    create_virtual_prefill_event = create_virtual_prefill_event_for_fp16

    if args.quantization == 'smooth_quant':
        create_virtual_decode_event = create_virtual_decode_event_for_smooth_quant
        create_virtual_prefill_event = None

    assert create_virtual_decode_event is not None, "create_virtual_decode_event is None"
    assert args.quantization == 'smooth_quant' or create_virtual_prefill_event is not None, "create_virtual_prefill_event is None"

    decode_start_kernels = finder.get_start_kernels(decode_start_kernel_id, decode_end_kernel_id, 'decode')
    prefill_start_kernels = finder.get_start_kernels(prefill_start_kernel_id, prefill_end_kernel_id, 'prefill')

    for kernel_start in tqdm(decode_start_kernels):
        create_virtual_decode_event(cursor, decode_kernel_mapping, kernel_start, decode_start_kernel_id, decode_end_kernel_id)

    if args.quantization != 'smooth_quant' and create_virtual_prefill_event is not None:
        for kernel_start in tqdm(prefill_start_kernels):
            create_virtual_prefill_event(cursor, prefill_kernel_mapping, kernel_start, prefill_start_kernel_id, prefill_end_kernel_id)

    avg_tpot, qkv_proj_latency, self_attn_latency, o_proj_latency, gate_up_proj_latency, down_proj_latency = analyze_decode_fp16(cursor)

    print("=" * 20)
    fprint("Avg. TPOT", avg_tpot)
    fprint("QKV PROJ", round(qkv_proj_latency * 32, 3))
    fprint("SELF ATTN PROJ", round(self_attn_latency * 32, 3))
    fprint("OUTPUT PROJ", round(o_proj_latency * 32, 3))
    fprint("GATE UP PROJ", round(gate_up_proj_latency * 32, 3))
    fprint("DOWN PROJ", round(down_proj_latency * 32, 3))
    print("=" * 20)

    avg_prefill, qkv_proj_latency, self_attn_latency, o_proj_latency, gate_up_proj_latency, down_proj_latency = analyze_prefill_fp16(cursor)

    fprint("Avg. PREFILL", avg_prefill)
    fprint("QKV PROJ", round(qkv_proj_latency * 32, 3))
    fprint("SELF ATTN PROJ", round(self_attn_latency * 32, 3))
    fprint("OUTPUT PROJ", round(o_proj_latency * 32, 3))
    fprint("GATE UP PROJ", round(gate_up_proj_latency * 32, 3))
    fprint("DOWN PROJ", round(down_proj_latency * 32, 3))
    print("=" * 20)

    connection.commit()

    connection.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("filename")
    parser.add_argument("--check-prefill-kernel", action='store_true')
    parser.add_argument("--check-decode-kernel", action='store_true')
    parser.add_argument("--quantization", choices=["smooth_quant", "FP16"], default="FP16")

    args = parser.parse_args()

    main(args)