import sys
import re
import argparse
import pandas as pd
from datetime import datetime
import os
import subprocess

def parse_header(filename):
    meta = {
        'Protocol': 'Unknown',
        'GPUs': 'Unknown',
        'Size': 'Unknown',
        'Algo': 'Unknown'
    }
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Protocol") and ":" in line:
                    meta['Protocol'] = line.split(':', 1)[1].strip()
                elif line.startswith("GPUs") and ":" in line:
                    meta['GPUs'] = line.split(':', 1)[1].strip()
                elif line.startswith("Data Size") and ":" in line:
                    val = line.split(':', 1)[1].strip()
                    parts = val.split()
                    if len(parts) >= 2:
                        meta['Size'] = parts[0] + parts[1]
                elif line.startswith("Algorithm") and ":" in line:
                    meta['Algo'] = line.split(':', 1)[1].strip()
                
                if "Compiling test" in line or "Running test" in line:
                    break
    except Exception as e:
        print(f"Warning: Could not parse header: {e}")
    return meta

def parse_log(filename):
    chunk_data = []
    slice_data = []
    tile_data = []
    pack_data = []
    
    # Regex patterns
    # NCCL_PROFILE_CHUNK,channelId,chunkId,OP,time
    p_chunk = re.compile(r"NCCL_PROFILE_CHUNK,(\d+),(\d+),(\w+),(\d+)")
    # NCCL_PROFILE_SLICE,channelId,chunkId,slice,TYPE,time
    p_slice = re.compile(r"NCCL_PROFILE_SLICE,(\d+),(\d+),(\d+),([\w_]+),(\d+)")
    # NCCL_PROFILE_TILE,channelId,chunkId,slice,tile,TYPE,time
    p_tile = re.compile(r"NCCL_PROFILE_TILE,(\d+),(\d+),(\d+),(\d+),([\w_]+),(\d+)")
    # NCCL_PROFILE_PACKS,GLOBAL|SHARED,unroll=...,pack=...,iters=...,load_total=...,store_total=...,load_avg=...,store_avg=...
    p_packs = re.compile(
        r"NCCL_PROFILE_PACKS,([A-Z_]+),"
        r"unroll=(\d+),pack=(\d+),iters=(\d+),"
        r"load_total=(\d+),store_total=(\d+),"
        r"load_avg=(\d+),store_avg=(\d+)"
    )

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            m_packs = p_packs.search(line)
            if m_packs:
                pack_data.append({
                    'MemoryPath': m_packs.group(1),
                    'Unroll': int(m_packs.group(2)),
                    'BytePerPack': int(m_packs.group(3)),
                    'Iterations': int(m_packs.group(4)),
                    'Load_Total_Cycles': int(m_packs.group(5)),
                    'Store_Total_Cycles': int(m_packs.group(6)),
                    'Load_Avg_Cycles': int(m_packs.group(7)),
                    'Store_Avg_Cycles': int(m_packs.group(8))
                })
                continue

            # Check most specific first to avoid partial matches if any
            m_tile = p_tile.search(line)
            if m_tile:
                tile_data.append({
                    'Channel': int(m_tile.group(1)),
                    'ChunkNum': int(m_tile.group(2)),
                    'SliceNum': int(m_tile.group(3)),
                    'TileNum': int(m_tile.group(4)),
                    'BlockType': m_tile.group(5),
                    'Time_Cycles': int(m_tile.group(6))
                })
                continue

            m_slice = p_slice.search(line)
            if m_slice:
                 slice_data.append({
                    'Channel': int(m_slice.group(1)),
                    'ChunkNum': int(m_slice.group(2)),
                    'SliceNum': int(m_slice.group(3)),
                    'BlockType': m_slice.group(4),
                    'Time_Cycles': int(m_slice.group(5))
                })
                 continue

            m_chunk = p_chunk.search(line)
            if m_chunk:
                chunk_data.append({
                    'Channel': int(m_chunk.group(1)),
                    'ChunkNum': int(m_chunk.group(2)),
                    'Operation': m_chunk.group(3),
                    'Time_Cycles': int(m_chunk.group(4))
                })
                continue
                
    return (
        pd.DataFrame(chunk_data),
        pd.DataFrame(slice_data),
        pd.DataFrame(tile_data),
        pd.DataFrame(pack_data),
    )

def default_output_base(meta, log_path):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"nccl_profile_{date_str}_{meta['Protocol']}_{meta['Size']}_{meta['GPUs']}gpu"
    log_dir = os.path.dirname(os.path.abspath(log_path))
    return os.path.join(log_dir, default_name).replace(" ", "")

def run_and_capture(args):
    if not args.run_script:
        return args.logfile

    if not args.output:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(os.getcwd(), f"nccl_profile_run_{date_str}")

    log_path = args.logfile if args.logfile else args.output + ".log"
    cmd = [args.run_script] + args.run_arg
    if not os.access(args.run_script, os.X_OK):
        cmd = ["bash", args.run_script] + args.run_arg

    env = os.environ.copy()
    for item in args.env:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        env[key] = value

    cwd = args.run_cwd if args.run_cwd else None

    print("Running command:")
    print("  " + " ".join(cmd))
    print(f"Logging to: {log_path}")

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"Run failed with exit code {rc}. Log saved to {log_path}")
    return log_path

def main():
    parser = argparse.ArgumentParser(description="Process NCCL Profile Logs (including TILE)")
    parser.add_argument("logfile", help="Path to log file", nargs='?')
    parser.add_argument("output", help="Optional output filename prefix", nargs='?')
    parser.add_argument("--freq", type=float, help="GPU frequency in GHz for time conversion (e.g., 1.98)", default=1.0)
    parser.add_argument("--run-script", help="Run a script first, capture stdout/stderr to a log, then process that log")
    parser.add_argument("--run-arg", action="append", default=[], help="Argument to pass to --run-script (repeatable)")
    parser.add_argument("--run-cwd", help="Working directory for --run-script")
    parser.add_argument("--env", action="append", default=[], help="Extra env for --run-script in KEY=VALUE form")
    
    args = parser.parse_args()
    if not args.logfile and not args.run_script:
        parser.error("either logfile or --run-script is required")

    try:
        log_path = run_and_capture(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Parsing {log_path}...")
    df_chunk, df_slice, df_tile, df_pack = parse_log(log_path)
    meta = parse_header(log_path)

    if df_chunk.empty and df_slice.empty and df_tile.empty and df_pack.empty:
        print("No profile data found in log.")
        sys.exit(1)

    print(f"Converting cycles to microseconds using {args.freq} GHz")
    time_col = 'Time_us'
    
    for df in [df_chunk, df_slice, df_tile]:
        if not df.empty:
            df['Time_us'] = df['Time_Cycles'] / (args.freq * 1000.0)

    if not df_pack.empty:
        df_pack['Load_Total_us'] = df_pack['Load_Total_Cycles'] / (args.freq * 1000.0)
        df_pack['Store_Total_us'] = df_pack['Store_Total_Cycles'] / (args.freq * 1000.0)
        df_pack['Load_Avg_us'] = df_pack['Load_Avg_Cycles'] / (args.freq * 1000.0)
        df_pack['Store_Avg_us'] = df_pack['Store_Avg_Cycles'] / (args.freq * 1000.0)

    dfs_to_save = {}

    # 1. Chunk Data
    if not df_chunk.empty:
        df_chunk = df_chunk.sort_values(by=['ChunkNum', 'Channel', 'Operation'])
        dfs_to_save['Chunk_Raw'] = df_chunk
        
        chunk_summary_ch = df_chunk.groupby(['Channel', 'ChunkNum', 'Operation'])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
        dfs_to_save['Chunk_Summary_Channel'] = chunk_summary_ch

        chunk_summary_global = df_chunk.groupby(['Operation'])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
        dfs_to_save['Chunk_Summary_Global'] = chunk_summary_global

    # 2. Slice Data
    if not df_slice.empty:
        # Keep tile-aggregate slice lines separate from regular slice profiling.
        # e.g. RECVSEND_TILE_WAIT_SUM, RECVSEND_TILE_STEP_SUM, ...
        is_tile_agg = df_slice['BlockType'].str.contains(r'_TILE_', regex=True)
        df_slice_regular = df_slice[~is_tile_agg].copy()
        df_slice_tileagg = df_slice[is_tile_agg].copy()

        if not df_slice_regular.empty:
            df_slice_regular = df_slice_regular.sort_values(by=['ChunkNum', 'SliceNum', 'Channel', 'BlockType'])
            dfs_to_save['Slice_Raw'] = df_slice_regular
            
            slice_summary_ch = df_slice_regular.groupby(['Channel', 'ChunkNum', 'SliceNum', 'BlockType'])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
            dfs_to_save['Slice_Summary_Channel'] = slice_summary_ch

            slice_summary_global = df_slice_regular.groupby(['BlockType'])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
            dfs_to_save['Slice_Summary_Global'] = slice_summary_global

        if not df_slice_tileagg.empty:
            df_slice_tileagg = df_slice_tileagg.sort_values(by=['ChunkNum', 'SliceNum', 'Channel', 'BlockType'])
            dfs_to_save['Slice_TileAgg_Raw'] = df_slice_tileagg

            slice_tileagg_ch = df_slice_tileagg.groupby(['Channel', 'ChunkNum', 'SliceNum', 'BlockType'])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
            dfs_to_save['Slice_TileAgg_Summary_Channel'] = slice_tileagg_ch

            slice_tileagg_global = df_slice_tileagg.groupby(['BlockType'])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
            dfs_to_save['Slice_TileAgg_Summary_Global'] = slice_tileagg_global

    # 3. Tile Data
    if not df_tile.empty:
        df_tile = df_tile.sort_values(by=['ChunkNum', 'SliceNum', 'TileNum', 'Channel', 'BlockType'])
        dfs_to_save['Tile_Raw'] = df_tile
        
        tile_summary_ch = df_tile.groupby(['Channel', 'ChunkNum', 'SliceNum', 'TileNum', 'BlockType'])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
        dfs_to_save['Tile_Summary_Channel'] = tile_summary_ch

        tile_summary_global = df_tile.groupby(['BlockType'])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
        dfs_to_save['Tile_Summary_Global'] = tile_summary_global

    # 4. Pack Timing Data
    if not df_pack.empty:
        df_pack = df_pack.sort_values(by=['MemoryPath', 'Unroll', 'BytePerPack'])
        dfs_to_save['Pack_Raw'] = df_pack

        pack_summary = df_pack.groupby(['MemoryPath', 'Unroll', 'BytePerPack']).agg(
            Samples=('Iterations', 'count'),
            Iterations_Total=('Iterations', 'sum'),
            Load_Total_Cycles_Mean=('Load_Total_Cycles', 'mean'),
            Load_Total_Cycles_Max=('Load_Total_Cycles', 'max'),
            Load_Total_Cycles_Min=('Load_Total_Cycles', 'min'),
            Store_Total_Cycles_Mean=('Store_Total_Cycles', 'mean'),
            Store_Total_Cycles_Max=('Store_Total_Cycles', 'max'),
            Store_Total_Cycles_Min=('Store_Total_Cycles', 'min'),
            Load_Avg_Cycles_Mean=('Load_Avg_Cycles', 'mean'),
            Load_Avg_Cycles_Max=('Load_Avg_Cycles', 'max'),
            Load_Avg_Cycles_Min=('Load_Avg_Cycles', 'min'),
            Store_Avg_Cycles_Mean=('Store_Avg_Cycles', 'mean'),
            Store_Avg_Cycles_Max=('Store_Avg_Cycles', 'max'),
            Store_Avg_Cycles_Min=('Store_Avg_Cycles', 'min'),
            Load_Total_us_Mean=('Load_Total_us', 'mean'),
            Store_Total_us_Mean=('Store_Total_us', 'mean'),
            Load_Avg_us_Mean=('Load_Avg_us', 'mean'),
            Store_Avg_us_Mean=('Store_Avg_us', 'mean'),
        ).reset_index()
        dfs_to_save['Pack_Summary'] = pack_summary


    out_base = args.output if args.output else default_output_base(meta, log_path)
    out_name = out_base
    if not out_name.endswith('.xlsx'):
        out_name += '.xlsx'
    
    out_name = out_name.replace(" ", "")
    
    print(f"Generated output filename: {out_name}")
    print(f"Writing results...")
    
    try:
        import openpyxl
        with pd.ExcelWriter(out_name, engine='openpyxl') as writer:
            for sheet_name, df in dfs_to_save.items():
                if len(sheet_name) > 31: # Excel restriction
                    sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Successfully saved to {out_name}")
    except (ModuleNotFoundError, ImportError):
        print("Error: 'openpyxl' module not found. Saving as separate CSV files.")
        base = out_name.replace('.xlsx', '')
        for sheet_name, df in dfs_to_save.items():
            fname = f"{base}_{sheet_name}.csv"
            df.to_csv(fname, index=False)
            print(f"Saved {fname}")

if __name__ == "__main__":
    main()
