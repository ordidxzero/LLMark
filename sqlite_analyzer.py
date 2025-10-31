import sqlite3, time, argparse
from tqdm import tqdm
# cursor.execute("SELECT event.id, ROUND((MAX(mapping.kernel_end)*1.0 - MIN(mapping.kernel_start)) / 1000000, 3) AS latency FROM (SELECT * FROM NVTX_EVENTS_WITH_KEY WHERE text='tpot') AS event JOIN KERNEL_RUNTIME_MAPPING AS mapping ON mapping.runtime_start >= event.start AND mapping.runtime_end <= event.end GROUP BY event.id ORDER BY event.id")

def has_mapping_table(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='KERNEL_RUNTIME_MAPPING'")
    result = cursor.fetchone()[0]

    return result != 0

def has_primary_key_in_nvtx(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='NVTX_EVENTS_WITH_KEY'")
    result = cursor.fetchone()[0]

    return result != 0

def has_bucket_index(cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='NVTX_EVENTS_WITH_KEY' AND name='idx_evt_bucket'")
    result = cursor.fetchone()[0]

    return result != 0

def get_computation_target(target: str):
    if target == 'attn':
        return "text LIKE '%attn%'"
    elif target == 'qkv':
        return "text LIKE '%qkv%'"
    elif target == 'gate_up':
        return "text LIKE '%gate_up%'"
    elif target == 'down':
        return "text LIKE '%down%'"
    elif target == 'o':
        return "text LIKE '%o_proj%'"
    elif target == 'prefill':
        return "text='prefill'"
    else:
        return "text='tpot'"
    

def get_connection(filename: str):
    connection = sqlite3.connect(filename)

    cursor = connection.cursor()

    return connection, cursor

def main(args: argparse.Namespace):
    connection, cursor = get_connection(args.filename)
    unit = '1000000.0' if args.unit == 'ms' else '1000.0'

    if not has_mapping_table(cursor):
        print("Create Mapping Table...")
        # 아래 쿼리는 CPU Runtime과 GPU Kernel을 연결시켜주는 쿼리임
        cursor.execute("CREATE TABLE KERNEL_RUNTIME_MAPPING AS SELECT kernel.start AS kernel_start, kernel.end AS kernel_end, runtime.start AS runtime_start, runtime.end AS runtime_end, kernel.correlationId, runtime.globalTid FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel JOIN CUPTI_ACTIVITY_KIND_RUNTIME runtime ON kernel.correlationId == runtime.correlationId")
        connection.commit()

    if not has_primary_key_in_nvtx(cursor):
        print("Create Primary Key...")
        cursor.execute("CREATE TABLE NVTX_EVENTS_WITH_KEY (id INTEGER PRIMARY KEY, start INTEGER NOT NULL, end INTEGER, eventType INTEGER NOT NULL, rangeId INTEGER, category INTEGER, color INTEGER, text TEXT, globalTid INTEGER, endGlobalTid INTEGER, textId INTEGER, domainId INTEGER)")
        cursor.execute("INSERT INTO NVTX_EVENTS_WITH_KEY (start, end, eventType, rangeId, category, color, text, globalTid, endGlobalTid, textId, domainId) SELECT start, end, eventType, rangeId, category, color, text, globalTid, endGlobalTid, textId, domainId FROM NVTX_EVENTS ORDER BY start ASC")
        connection.commit()

    if not has_bucket_index(cursor):
        print("Create Bucket Index...")
        cursor.execute("ALTER TABLE KERNEL_RUNTIME_MAPPING ADD COLUMN runtime_start_bucket INTEGER GENERATED ALWAYS AS (FLOOR(runtime_start / 1000000)) VIRTUAL")
        cursor.execute("ALTER TABLE KERNEL_RUNTIME_MAPPING ADD COLUMN runtime_end_bucket INTEGER GENERATED ALWAYS AS (FLOOR(runtime_end / 1000000)) VIRTUAL")
        cursor.execute("ALTER TABLE NVTX_EVENTS_WITH_KEY ADD COLUMN start_bucket INTEGER GENERATED ALWAYS AS (FLOOR(start / 1000000)) VIRTUAL")
        cursor.execute("ALTER TABLE NVTX_EVENTS_WITH_KEY ADD COLUMN end_bucket INTEGER GENERATED ALWAYS AS (FLOOR(end / 1000000)) VIRTUAL")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_map_bucket ON KERNEL_RUNTIME_MAPPING (runtime_start_bucket, runtime_end_bucket, runtime_start, runtime_end, kernel_start, kernel_end)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evt_bucket ON NVTX_EVENTS_WITH_KEY (id, start_bucket, end_bucket)")
        connection.commit()

    target = get_computation_target(args.target)
    # QUERY = f"WITH event AS (SELECT id, start, end FROM NVTX_EVENTS_WITH_KEY WHERE {target} AND id > {args.skip_offset}) SELECT e.id, ROUND((MAX(m.kernel_end) - MIN(m.kernel_start)) / {unit}, 3) AS latency FROM event AS e JOIN KERNEL_RUNTIME_MAPPING m ON m.runtime_start >= e.start AND m.runtime_end <= e.end GROUP BY e.id ORDER BY e.id"
    QUERY = f" WITH event AS (SELECT id, start, end, start_bucket, end_bucket FROM NVTX_EVENTS_WITH_KEY INDEXED BY idx_evt_bucket WHERE {target} AND id > {args.skip_offset}) SELECT e.id, ROUND((MAX(m.kernel_end) - MIN(m.kernel_start)) / {unit}, 3) AS latency FROM event AS e JOIN KERNEL_RUNTIME_MAPPING AS m INDEXED BY idx_map_bucket ON m.runtime_start_bucket BETWEEN e.start_bucket AND e.end_bucket AND m.runtime_end_bucket BETWEEN e.start_bucket AND e.end_bucket AND m.runtime_start >= e.start AND m.runtime_end <= e.end GROUP BY e.id"

    cursor.execute(f"SELECT COUNT(*) FROM NVTX_EVENTS_WITH_KEY WHERE {target} AND id > {args.skip_offset}")
    count = cursor.fetchone()[0]


    events = []
    start = time.perf_counter_ns()

    pass_upper_bound = min(args.max, 0) < 0
    pass_lower_bound = min(args.min, 0) < 0
    

    for row in tqdm(cursor.execute(QUERY), total=count):
        id, latency = row
        
        if pass_upper_bound:
            if pass_lower_bound:
                events.append(latency)
            else:
                if latency > args.min:
                    events.append(latency)

            continue

        if pass_lower_bound:
            if latency < args.max:
                events.append(latency)
            
            continue

        if latency < args.max and latency > args.min:
            events.append(latency)
    
    print(f"Avg. {args.target} Latency: {sum(events) / (len(events) / args.group_size):.2f}")
    print(f"Total {args.target} Latency: {sum(events):.2f}")
    end = time.perf_counter_ns()

    print("Time: ", (end - start) / 1000000000)

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

    # 1k: 100959 / 38.13s / 2508
    # 2k: 153935 / 67.16s / 4780
    # 4k: 140423 / 91.79s / 7257
    # 8k: 145577 / 171.81s / 16272