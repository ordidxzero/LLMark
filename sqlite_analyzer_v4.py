import argparse, re, sqlite3
from llmark.sqlite import get_prefill_events, get_prefill_kernels
from llmark.sqlite.logger import KernelLogger
from llmark.sqlite.calculator import KernelCalculator
from llmark.sqlite.smooth_quant import SmoothQuantStateMachine
from llmark.sqlite.float16 import Float16StateMachine, FP16_DECODE_KERNEL_V2, FP16_PREFILL_KERNEL_V2
from llmark.sqlite.gptq_marlin import GPTQMarlinStateMachine

def connect(filename: str):
    connection = sqlite3.connect(filename)
    cursor = connection.cursor()

    return cursor, connection

def has_dram_bdw_seg(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='DRAM_BANDWIDTH_SEG'")
    result = cursor.fetchone()[0]

    return result != 0

def has_sm_active_seg(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='SM_ACTIVE_SEG'")
    result = cursor.fetchone()[0]

    return result != 0

def has_compute_warp_seg(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='COMPUTE_WARP_SEG'")
    result = cursor.fetchone()[0]

    return result != 0

def has_unallocated_warp_seg(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='UNALLOCATED_WARP_SEG'")
    result = cursor.fetchone()[0]

    return result != 0

def has_tensor_active_seg(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='TENSOR_ACTIVE_SEG'")
    result = cursor.fetchone()[0]

    return result != 0

def has_seg_index_in_bdw(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='DRAM_BANDWIDTH_SEG' AND name='idx_bdw_bucket'")
    result = cursor.fetchone()[0]

    return result != 0

def has_seg_index_in_sm(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='SM_ACTIVE_SEG' AND name='idx_sm_bucket'")
    result = cursor.fetchone()[0]

    return result != 0

def has_seg_index_in_compute_warp(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='COMPUTE_WARP_SEG' AND name='idx_compute_warp_bucket'")
    result = cursor.fetchone()[0]

    return result != 0

def has_seg_index_in_unallocated_warp(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='UNALLOCATED_WARP_SEG' AND name='idx_unallocated_warp_bucket'")
    result = cursor.fetchone()[0]

    return result != 0

def has_seg_index_in_tensor_active(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='TENSOR_ACTIVE_SEG' AND name='idx_tensor_bucket'")
    result = cursor.fetchone()[0]

    return result != 0

def init(cursor: sqlite3.Cursor, connection: sqlite3.Connection):
    cursor.execute("CREATE TABLE IF NOT EXISTS KERNEL_WITH_INDEX AS SELECT start, end, demangledName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_kernel_start ON KERNEL_WITH_INDEX(start)")


    QUERY = "SELECT metricId FROM TARGET_INFO_GPU_METRICS WHERE metricName Like '%Dram Read%'"
    cursor.execute(QUERY)
    metric_id = cursor.fetchone()[0]
    if not has_dram_bdw_seg(cursor):
        cursor.execute("CREATE TABLE DRAM_BANDWIDTH_SEG (start INTEGER NOT NULL, end INTEGER NOT NULL, value REAL NOT NULL, PRIMARY KEY (start)) WITHOUT ROWID")
        cursor.execute(f"INSERT INTO DRAM_BANDWIDTH_SEG (start, end, value) WITH base AS (SELECT timestamp AS start, LEAD(timestamp) OVER (ORDER BY timestamp) AS end, value FROM GPU_METRICS WHERE metricId = {metric_id}) SELECT start, end, value FROM base WHERE end IS NOT NULL AND (end - start) <= 100000000 ORDER BY start")
    if not has_seg_index_in_bdw(cursor):
        cursor.execute("ALTER TABLE DRAM_BANDWIDTH_SEG ADD COLUMN seg_start_bucket INTEGER GENERATED ALWAYS AS (start / 1000000) VIRTUAL")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bdw_bucket ON DRAM_BANDWIDTH_SEG(seg_start_bucket, start, end, value)")
    cursor.execute("DELETE FROM DRAM_BANDWIDTH_SEG WHERE end - start > 100000000")

    QUERY = "SELECT metricId FROM TARGET_INFO_GPU_METRICS WHERE metricName Like '%SMs Active%'"
    cursor.execute(QUERY)
    metric_id = cursor.fetchone()[0]
    if not has_sm_active_seg(cursor):
        cursor.execute("CREATE TABLE SM_ACTIVE_SEG (start INTEGER NOT NULL, end INTEGER NOT NULL, value REAL NOT NULL, PRIMARY KEY (start)) WITHOUT ROWID")
        cursor.execute(f"INSERT INTO SM_ACTIVE_SEG (start, end, value) WITH base AS (SELECT timestamp AS start, LEAD(timestamp) OVER (ORDER BY timestamp) AS end, value FROM GPU_METRICS WHERE metricId = {metric_id}) SELECT start, end, value FROM base WHERE end IS NOT NULL AND (end - start) <= 100000000 ORDER BY start")
    if not has_seg_index_in_sm(cursor):
        cursor.execute("ALTER TABLE SM_ACTIVE_SEG ADD COLUMN seg_start_bucket INTEGER GENERATED ALWAYS AS (start / 1000000) VIRTUAL")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sm_bucket ON SM_ACTIVE_SEG(seg_start_bucket, start, end, value)")
    cursor.execute("DELETE FROM SM_ACTIVE_SEG WHERE end - start > 100000000")

    QUERY = "SELECT metricId FROM TARGET_INFO_GPU_METRICS WHERE metricName LIKE '%Compute Warps in Flight%' AND metricName LIKE '%Throughput%'"
    cursor.execute(QUERY)
    metric_id = cursor.fetchone()[0]
    if not has_compute_warp_seg(cursor):
        cursor.execute("CREATE TABLE COMPUTE_WARP_SEG (start INTEGER NOT NULL, end INTEGER NOT NULL, value REAL NOT NULL, PRIMARY KEY (start)) WITHOUT ROWID")
        cursor.execute(f"INSERT INTO COMPUTE_WARP_SEG (start, end, value) WITH base AS (SELECT timestamp AS start, LEAD(timestamp) OVER (ORDER BY timestamp) AS end, value FROM GPU_METRICS WHERE metricId = {metric_id}) SELECT start, end, value FROM base WHERE end IS NOT NULL AND (end - start) <= 100000000 ORDER BY start")
    if not has_seg_index_in_compute_warp(cursor):
        cursor.execute("ALTER TABLE COMPUTE_WARP_SEG ADD COLUMN seg_start_bucket INTEGER GENERATED ALWAYS AS (start / 1000000) VIRTUAL")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_compute_warp_bucket ON COMPUTE_WARP_SEG(seg_start_bucket, start, end, value)")
    cursor.execute("DELETE FROM COMPUTE_WARP_SEG WHERE end - start > 100000000")

    QUERY = "SELECT metricId FROM TARGET_INFO_GPU_METRICS WHERE metricName LIKE '%Unallocated Warps in Active%' AND metricName LIKE '%Throughput%'"
    cursor.execute(QUERY)
    metric_id = cursor.fetchone()[0]
    if not has_unallocated_warp_seg(cursor):
        cursor.execute("CREATE TABLE UNALLOCATED_WARP_SEG (start INTEGER NOT NULL, end INTEGER NOT NULL, value REAL NOT NULL, PRIMARY KEY (start)) WITHOUT ROWID")
        cursor.execute(f"INSERT INTO UNALLOCATED_WARP_SEG (start, end, value) WITH base AS (SELECT timestamp AS start, LEAD(timestamp) OVER (ORDER BY timestamp) AS end, value FROM GPU_METRICS WHERE metricId = {metric_id}) SELECT start, end, value FROM base WHERE end IS NOT NULL AND (end - start) <= 100000000 ORDER BY start")
    if not has_seg_index_in_unallocated_warp(cursor):
        cursor.execute("ALTER TABLE UNALLOCATED_WARP_SEG ADD COLUMN seg_start_bucket INTEGER GENERATED ALWAYS AS (start / 1000000) VIRTUAL")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_unallocated_warp_bucket ON UNALLOCATED_WARP_SEG(seg_start_bucket, start, end, value)")
    cursor.execute("DELETE FROM UNALLOCATED_WARP_SEG WHERE end - start > 100000000")

    QUERY = "SELECT metricId FROM TARGET_INFO_GPU_METRICS WHERE metricName LIKE '%Tensor Active%' AND metricName LIKE '%Throughput%'"
    cursor.execute(QUERY)
    metric_id = cursor.fetchone()[0]
    if not has_tensor_active_seg(cursor):
        cursor.execute("CREATE TABLE TENSOR_ACTIVE_SEG (start INTEGER NOT NULL, end INTEGER NOT NULL, value REAL NOT NULL, PRIMARY KEY (start)) WITHOUT ROWID")
        cursor.execute(f"INSERT INTO TENSOR_ACTIVE_SEG (start, end, value) WITH base AS (SELECT timestamp AS start, LEAD(timestamp) OVER (ORDER BY timestamp) AS end, value FROM GPU_METRICS WHERE metricId = {metric_id}) SELECT start, end, value FROM base WHERE end IS NOT NULL AND (end - start) <= 100000000 ORDER BY start")
    if not has_seg_index_in_tensor_active(cursor):
        cursor.execute("ALTER TABLE TENSOR_ACTIVE_SEG ADD COLUMN seg_start_bucket INTEGER GENERATED ALWAYS AS (start / 1000000) VIRTUAL")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tensor_bucket ON TENSOR_ACTIVE_SEG(seg_start_bucket, start, end, value)")
    cursor.execute("DELETE FROM TENSOR_ACTIVE_SEG WHERE end - start > 100000000")
    connection.commit()
    return

def get_max_num_seqs(filename: str):
    nums = re.search(r"max-num-seqs_(\d+)_num-prompts_(\d+)", filename)
    assert nums is not None, "error"
    max_num_seqs = int(nums.group(1))
    num_prompts = int(nums.group(2))
    return max_num_seqs, int(num_prompts / max_num_seqs)


def main(args: argparse.Namespace):
    cursor, connection = connect(args.filename)
    max_num_seqs, num_prefill = get_max_num_seqs(args.filename)

    init(cursor, connection)
    quantization = 'float16'
    if 'smooth_quant' in args.filename:
        quantization = 'smooth_quant'
    elif 'gptq_marlin' in args.filename:
        quantization = 'gptq_marlin'
    else:
        pass


    prefill_events = get_prefill_events(cursor, max_num_seqs, num_prefill)
    prefill_kernels = get_prefill_kernels(cursor, quantization, prefill_events=prefill_events)
    
    if args.show_kernels:
        logger = KernelLogger(cursor=cursor, prefill_kernels=prefill_kernels)
        if args.phase == 'decode':
            logger.print_decode_kernels()
        if args.phase == 'prefill':
            logger.print_prefill_kernels()
        return
    
    if quantization == 'float16':
        if args.phase == 'decode':
            state_machine = Float16StateMachine(cursor, predefined_kernel=FP16_DECODE_KERNEL_V2.get(max_num_seqs)) # type: ignore
        else:
            state_machine = Float16StateMachine(cursor, predefined_kernel=FP16_PREFILL_KERNEL_V2.get(max_num_seqs)) # type: ignore
    elif quantization == 'smooth_quant':
        state_machine = SmoothQuantStateMachine(phase=args.phase)
    else:
        state_machine = GPTQMarlinStateMachine(phase=args.phase)

    calculator = KernelCalculator(cursor, connection, prefill_kernels=prefill_kernels, workload='decode_heavy')
    # calculator.insert_ops(state_machine)
    
    if args.show_dram:
        calculator.calc_dram_bandwidth(args.phase, state_machine, max_num_seqs, quantization)
        return

    if args.phase == 'decode':
        calculator.calc_decode_kernels(state_machine)
    else:
        calculator.calc_prefill_kernels(state_machine)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("filename")
    parser.add_argument("phase", choices=['decode', 'prefill'])
    parser.add_argument("--show-kernels", action='store_true')
    parser.add_argument("--show-dram", action='store_true')

    args = parser.parse_args()

    main(args)