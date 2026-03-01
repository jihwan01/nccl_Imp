import sys
import re
import argparse
import pandas as pd
from datetime import datetime
import os

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
    
    # Regex patterns
    # NCCL_PROFILE_CHUNK,channelId,chunkId,OP,time
    p_chunk = re.compile(r"NCCL_PROFILE_CHUNK,(\d+),(\d+),(\w+),(\d+)")
    # NCCL_PROFILE_SLICE,channelId,chunkId,slice,TYPE,time
    p_slice = re.compile(r"NCCL_PROFILE_SLICE,(\d+),(\d+),(\d+),([\w_]+),(\d+)")
    # NCCL_PROFILE_TILE,channelId,chunkId,slice,tile,TYPE,time
    p_tile = re.compile(r"NCCL_PROFILE_TILE,(\d+),(\d+),(\d+),(\d+),([\w_]+),(\d+)")

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
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
                
    return pd.DataFrame(chunk_data), pd.DataFrame(slice_data), pd.DataFrame(tile_data)

def main():
    parser = argparse.ArgumentParser(description="Process NCCL Profile Logs (including TILE)")
    parser.add_argument("logfile", help="Path to log file")
    parser.add_argument("output", help="Optional output filename prefix", nargs='?')
    parser.add_argument("--freq", type=float, help="GPU frequency in GHz for time conversion (e.g., 1.98)", default=1.0)
    
    args = parser.parse_args()

    print(f"Parsing {args.logfile}...")
    df_chunk, df_slice, df_tile = parse_log(args.logfile)
    meta = parse_header(args.logfile)

    if df_chunk.empty and df_slice.empty and df_tile.empty:
        print("No profile data found in log.")
        sys.exit(1)

    print(f"Converting cycles to microseconds using {args.freq} GHz")
    time_col = 'Time_us'
    
    for df in [df_chunk, df_slice, df_tile]:
        if not df.empty:
            df['Time_us'] = df['Time_Cycles'] / (args.freq * 1000.0)

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
        df_slice = df_slice.sort_values(by=['ChunkNum', 'SliceNum', 'Channel', 'BlockType'])
        dfs_to_save['Slice_Raw'] = df_slice
        
        slice_summary_ch = df_slice.groupby(['Channel', 'ChunkNum', 'SliceNum', 'BlockType'])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
        dfs_to_save['Slice_Summary_Channel'] = slice_summary_ch

        slice_summary_global = df_slice.groupby(['BlockType'])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
        dfs_to_save['Slice_Summary_Global'] = slice_summary_global

    # 3. Tile Data
    if not df_tile.empty:
        df_tile = df_tile.sort_values(by=['ChunkNum', 'SliceNum', 'TileNum', 'Channel', 'BlockType'])
        dfs_to_save['Tile_Raw'] = df_tile
        
        tile_summary_ch = df_tile.groupby(['Channel', 'ChunkNum', 'SliceNum', 'TileNum', 'BlockType'])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
        dfs_to_save['Tile_Summary_Channel'] = tile_summary_ch

        tile_summary_global = df_tile.groupby(['BlockType'])[time_col].agg(['mean', 'max', 'min', 'count']).reset_index()
        dfs_to_save['Tile_Summary_Global'] = tile_summary_global


    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"nccl_profile_{date_str}_{meta['Protocol']}_{meta['Size']}_{meta['GPUs']}gpu.xlsx"
    # Default output directory follows the input log location, not current cwd.
    log_dir = os.path.dirname(os.path.abspath(args.logfile))
    out_name = args.output if args.output else os.path.join(log_dir, default_name)
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
