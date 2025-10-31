# =================================================
# TABLE 분석
# =================================================

# ============================================
# NVTX_EVENTS
# 
# start (INT, ns)
# end (INT, ns)
# eventType (INT, REFERENCE ENUM_NSYS_EVENT_TYPE)
# text (TEXT, Explicit name/text)
# globalTid (INT, Serialized GlobalId)
# domainId (INT, User-controlled ID that can be used to group events)

# ============================================

# ============================================
# CUPTI_ACTIVITY_KIND_KERNEL
# 
# start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
# end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
# deviceId                    INTEGER   NOT NULL,                    -- Device ID.
# contextId                   INTEGER   NOT NULL,                    -- Context ID.
# greenContextId              INTEGER,                               -- Green context ID.
# streamId                    INTEGER   NOT NULL,                    -- Stream ID.
# correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
# globalPid                   INTEGER,                               -- Serialized GlobalId.
# demangledName               INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Kernel function name w/ templates <-- 커널 이름
# shortName                   INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Base kernel function name
# mangledName                 INTEGER,                               -- REFERENCES StringIds(id) -- Raw C++ mangled kernel function name
# launchType                  INTEGER,                               -- REFERENCES ENUM_CUDA_KERNEL_LAUNCH_TYPE(id)
# cacheConfig                 INTEGER,                               -- REFERENCES ENUM_CUDA_FUNC_CACHE_CONFIG(id)
# registersPerThread          INTEGER   NOT NULL,                    -- Number of registers required for each thread executing the kernel.
# gridX                       INTEGER   NOT NULL,                    -- X-dimension grid size.
# gridY                       INTEGER   NOT NULL,                    -- Y-dimension grid size.
# gridZ                       INTEGER   NOT NULL,                    -- Z-dimension grid size.
# blockX                      INTEGER   NOT NULL,                    -- X-dimension block size.
# blockY                      INTEGER   NOT NULL,                    -- Y-dimension block size.
# blockZ                      INTEGER   NOT NULL,                    -- Z-dimension block size.
# staticSharedMemory          INTEGER   NOT NULL,                    -- Static shared memory allocated for the kernel (B).
# dynamicSharedMemory         INTEGER   NOT NULL,                    -- Dynamic shared memory reserved for the kernel (B).
# localMemoryPerThread        INTEGER   NOT NULL,                    -- Amount of local memory reserved for each thread (B).
# localMemoryTotal            INTEGER   NOT NULL,                    -- Total amount of local memory reserved for the kernel (B).
# gridId                      INTEGER   NOT NULL,                    -- Unique grid ID of the kernel assigned at runtime.
# sharedMemoryExecuted        INTEGER,                               -- Shared memory size set by the driver.
# graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
# sharedMemoryLimitConfig     INTEGER,                               -- REFERENCES ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG(id)
# qmdBulkReleaseDone          INTEGER,                               -- QMD bulk release done timestamp from CWD events.
# qmdPreexitDone              INTEGER,                               -- QMD pre-exit done timestamp from CWD events.
# qmdLastCtaDone              INTEGER,                               -- QMD last CTA done timestamp from CWD events.
# graphId                     INTEGER,                               -- Kernel graph ID.
# clusterX                    INTEGER,                               -- Cluster X dimension.
# clusterY                    INTEGER,                               -- Cluster Y dimension.
# clusterZ                    INTEGER,                               -- Cluster Z dimension.
# clusterSchedulingPolicy     INTEGER,                               -- Cluster scheduling policy.
# maxPotentialClusterSize     INTEGER,                               -- Maximum potential cluster size.
# maxActiveClusters           INTEGER                                -- Maximum active clusters.
# ============================================

# ============================================
# CUPTI_ACTIVITY_KIND_RUNTIME
# 
# start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
# end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
# eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
# globalTid                   INTEGER,                               -- Serialized GlobalId.
# correlationId               INTEGER,                               -- ID used to identify events that this function call has triggered.
# nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
# returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
# callchainId                 INTEGER                                -- REFERENCES CUDA_CALLCHAINS(id)
# ============================================


# =================================================
# =================================================

import sqlite3, time, argparse
from typing import List, Tuple, Dict
from tqdm import tqdm

class SmoothQuantKernelMapping:
    pass

class KernelMapping:
    qkv_proj: int
    self_attn: List[int]
    o_or_down_proj: int
    gate_up_proj_or_lm_head: int

    def __init__(self):
        self.qkv_proj = -1
        self.self_attn = []
        self.o_or_down_proj = -1
        self.gate_up_proj_or_lm_head = -1

    def set_qkv_proj(self, id: int):
        self.qkv_proj = id

    def add_self_attn(self, id: int):
        assert len(self.self_attn) + 1 <= 3, "Error"

        self.self_attn.append(id)
    
    def set_o_or_down_proj(self, id: int):
        self.o_or_down_proj = id

    def set_gate_up_proj_or_lm_head(self, id: int):
        self.gate_up_proj_or_lm_head = id

    def get_name(self, id: int, is_o_proj: bool, is_lm_head: bool):
        if self.qkv_proj == id:
            return "qkv_proj"

        if id in self.self_attn:
            if id == self.self_attn[-1]:
                return "self_attn_end"
            return "self_attn"

        if self.o_or_down_proj == id:
            if is_o_proj: return 'o_proj'
            else: return "down_proj"
        
        if self.gate_up_proj_or_lm_head == id:
            if is_lm_head: return 'lm_head'
            return 'gate_up_proj'
        
        return None

# 1. Prefill NVTX로부터 nvtx text - kernel name 매핑하기
# 1-1. Prefill 하나 불러오기
# SELECT * FROM NVTX_EVENTS AS nvtx WHERE nvtx.text LIKE '%PREFILL%' LIMIT 1 OFFSET 1;
# 1-2. Prefill 안에서 5가지 연산에 대한 이름 가져오기 (qkv, self, o, gate_up, down)
# SELECT nvtx.text, nvtx.start, nvtx.end FROM NVTX_EVENTS AS nvtx WHERE nvtx.start > ... AND nvtx.end < ... LIMIT 5;
# nameId == '122'인 것들 제외하기

# ================================================
# Intialization
# ================================================
def connect(filename: str):
    connection = sqlite3.connect(filename)
    cursor = connection.cursor()

    return cursor, connection

def init_virtual_nvtx_events(cursor: sqlite3.Cursor, connection: sqlite3.Connection):
    cursor.execute("CREATE TABLE IF NOT EXISTS VIRTUAL_NVTX_OPS (pid INTEGER PRIMARY KEY, name TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL)")
    cursor.execute("CREATE TABLE IF NOT EXISTS VIRTUAL_NVTX_EVENTS (pid INTEGER PRIMARY KEY, name TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL)")
    connection.commit()
    return

kernel_mapping = KernelMapping()
def init_kernel_mapping(cursor: sqlite3.Cursor):
    global kernel_mapping
    QUERY = "SELECT id FROM StringIds WHERE value = 'void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_64x6_tn_align8>(T1::Params)'"
    cursor.execute(QUERY)
    qkv_proj_id = cursor.fetchone()[0]
    kernel_mapping.set_qkv_proj(qkv_proj_id)
    
    QUERY = "SELECT id FROM StringIds WHERE value LIKE '%void vllm::reshape_and_cache_flash_kernel%'"
    cursor.execute(QUERY)
    self_attn_id_1 = cursor.fetchone()[0]
    kernel_mapping.add_self_attn(self_attn_id_1)
    
    QUERY = "SELECT id FROM StringIds WHERE value LIKE '%void flash_fwd_splitkv_kernel%'"
    cursor.execute(QUERY)
    self_attn_id_2 = cursor.fetchone()[0]
    kernel_mapping.add_self_attn(self_attn_id_2)
    
    QUERY = "SELECT id FROM StringIds WHERE value LIKE '%void flash_fwd_splitkv_combine_kernel%'"
    cursor.execute(QUERY)
    self_attn_id_3 = cursor.fetchone()[0]
    kernel_mapping.add_self_attn(self_attn_id_3)

    QUERY = "SELECT id FROM StringIds WHERE value = 'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn'"
    cursor.execute(QUERY)
    o_or_down_proj_id = cursor.fetchone()[0]
    kernel_mapping.set_o_or_down_proj(o_or_down_proj_id)

    QUERY = "SELECT id FROM StringIds WHERE value = 'ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn'"
    cursor.execute(QUERY)
    gate_up_proj_id = cursor.fetchone()[0]
    kernel_mapping.set_gate_up_proj_or_lm_head(gate_up_proj_id)

    

def has_kernel_with_index_table(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='KERNEL_WITH_INDEX'")
    result = cursor.fetchone()[0]

    return result != 0

def has_bucket_index(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='KERNEL_WITH_INDEX' AND name='kernel_time_bucket'")
    result = cursor.fetchone()[0]

    return result != 0

# ================================================
# ================================================

# ================================================
# TPOT의 시작과 끝에 해당하는 커널의 아이디 검색하기
# ================================================
def find_tpot_boundary_kernels(cursor: sqlite3.Cursor) -> Tuple[int, int]:
    start_kernel_name = 'void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)'
    end_kernel_name = 'void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, (int)4>>(T3)'

    QUERY = f"SELECT id FROM StringIds WHERE value = '{start_kernel_name}'"
    cursor.execute(QUERY)
    start_kernel_id = cursor.fetchone()
    assert start_kernel_id is not None, "Not Found Kernel ID (start_kernel_id)"

    QUERY = f"SELECT id FROM StringIds WHERE value = '{end_kernel_name}'"
    cursor.execute(QUERY)
    end_kernel_id = cursor.fetchone()
    assert end_kernel_id is not None, "Not Found Kernel ID (end_kernel_id)"

    return start_kernel_id[0], end_kernel_id[0]

def find_tpot_start_kernels(cursor: sqlite3.Cursor, start_kernel_id: int, time_offset: int = 0):
    QUERY = f"SELECT start FROM KERNEL_WITH_INDEX WHERE demangledName = '{start_kernel_id}' AND start >= {time_offset} ORDER BY start ASC"
    cursor.execute(QUERY)

    start_kernels = cursor.fetchall()
    

    return start_kernels

def create_virtual_nvtx_event(cursor: sqlite3.Cursor, connection: sqlite3.Connection, time_offsetoffset: int, start_kernel_id: int, end_kernel_id: int):
    QUERY = f"SELECT start, end, demangledName FROM KERNEL_WITH_INDEX WHERE start >= {time_offsetoffset} ORDER BY start ASC LIMIT 400"
    cursor.execute(QUERY)

    kernels = cursor.fetchall()

    event_start_time = 0
    attn_start_time = 0
    is_o_proj = True
    is_lm_head = False

    i = 0
    for kernel in kernels:
        kernel_start, kernel_end, kernel_id = kernel
        if kernel_id == start_kernel_id:
            event_start_time = kernel_start
            continue
        if kernel_id == end_kernel_id:
            QUERY = f"INSERT INTO VIRTUAL_NVTX_EVENTS (name, start, end) VALUES ('TPOT', {event_start_time}, {kernel_end})"
            cursor.execute(QUERY)
            break

        is_lm_head = i == 32
        name = kernel_mapping.get_name(kernel_id, is_o_proj, is_lm_head)
        if name is None:
            continue

        if name == 'qkv_proj':
            QUERY = f"INSERT INTO VIRTUAL_NVTX_OPS (name, start, end) VALUES ('{name}', {kernel_start}, {kernel_end})"
            cursor.execute(QUERY)
        elif "self_attn" in name:
            if attn_start_time == 0:
                attn_start_time = kernel_start
            
            if name == 'self_attn_end':
                QUERY = f"INSERT INTO VIRTUAL_NVTX_OPS (name, start, end) VALUES ('self_attn', {attn_start_time}, {kernel_end})"
                cursor.execute(QUERY)
                # Something..
            
        elif name == 'o_proj':
            is_o_proj = False
            QUERY = f"INSERT INTO VIRTUAL_NVTX_OPS (name, start, end) VALUES ('{name}', {kernel_start}, {kernel_end})"
            cursor.execute(QUERY)
        elif name == 'gate_up_proj':
            QUERY = f"INSERT INTO VIRTUAL_NVTX_OPS (name, start, end) VALUES ('{name}', {kernel_start}, {kernel_end})"
            cursor.execute(QUERY)
        elif name == 'down_proj':
            is_o_proj = True
            QUERY = f"INSERT INTO VIRTUAL_NVTX_OPS (name, start, end) VALUES ('{name}', {kernel_start}, {kernel_end})"
            cursor.execute(QUERY)
            i+=1
        elif name == 'lm_head':
            QUERY = f"INSERT INTO VIRTUAL_NVTX_OPS (name, start, end) VALUES ('{name}', {kernel_start}, {kernel_end})"
            cursor.execute(QUERY)
        else:
            raise Exception("NotImplemented")
        
# ================================================
# ================================================

def create_virtual_nvtx_for_fp16(cursor: sqlite3.Cursor, connection: sqlite3.Connection, time_offset: int = 0):
    start_kernel_id, end_kernel_id = find_tpot_boundary_kernels(cursor)

    QUERY = f"SELECT start FROM KERNEL_WITH_INDEX WHERE demangledName = '{start_kernel_id}' AND start >= {time_offset} ORDER BY start ASC"
    cursor.execute(QUERY)

    start_kernels = cursor.fetchall()

    for kernel in tqdm(start_kernels):
        kernel_start = kernel[0]
        create_virtual_nvtx_event(cursor, connection, kernel_start, start_kernel_id, end_kernel_id)

    connection.commit()


def main(args: argparse.Namespace):
    cursor, connection = connect(args.filename)

    init_kernel_mapping(cursor)
    init_virtual_nvtx_events(cursor, connection)

    if not has_kernel_with_index_table(cursor):
        cursor.execute("CREATE TABLE KERNEL_WITH_INDEX AS SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL")
        connection.commit()

    if not has_bucket_index(cursor):
        print("Create Bucket Index...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kernel_start ON KERNEL_WITH_INDEX(start)")
        connection.commit()

    create_virtual_nvtx_for_fp16(cursor, connection)

    connection.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("filename")
    parser.add_argument("--target", choices=['prefill', 'tpot', 'qkv', 'attn', 'o', 'gate_up', 'down'])
    parser.add_argument("--skip-offset", type=int, default=0)
    parser.add_argument("--min", type=int, default=-1)
    parser.add_argument("--max", type=int, default=-1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--unit", choices=['ms', 'us'], default='ms')

    args = parser.parse_args()

    main(args)