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
    ll_data = []
    ll128_data = []
    ll_detail_data = []
    ll128_detail_data = []
    
    # Regex patterns
    # NCCL_PROFILE_CHUNK,channelId,chunkId,OP,time
    p_chunk = re.compile(r"NCCL_PROFILE_CHUNK,(\d+),(\d+),(\w+),(\d+)")
    # NCCL_PROFILE_SLICE,channelId,chunkId,slice,TYPE,time
    p_slice = re.compile(r"NCCL_PROFILE_SLICE,(\d+),(\d+),(\d+),([\w_]+),(\d+)")
    # NCCL_PROFILE_TILE,channelId,chunkId,slice,tile,TYPE,time
    p_tile = re.compile(r"NCCL_PROFILE_TILE,(\d+),(\d+),(\d+),(\d+),([\w_]+),(\d+)")
    # NCCL_PROFILE_LL,channelId,chunkId,OP,load,send,sync,total,elems
    p_ll = re.compile(r"NCCL_PROFILE_LL,(\d+),(-?\d+),([\w_]+),(\d+),(\d+),(\d+),(\d+),(\d+)")
    # NCCL_PROFILE_LL128,channelId,chunkId,OP,load,send,sync,total,elems
    p_ll128 = re.compile(r"NCCL_PROFILE_LL128,(\d+),(-?\d+),([\w_]+),(\d+),(\d+),(\d+),(\d+),(\d+)")
    # NCCL_PROFILE_LL_DETAIL,channel,chunk,op,loadBegin,loadFinish,dstStore,sendStore,waitSendCredit,waitSendFifo,waitSendBarrier,waitSendPolls,recvBegin,recvBeginLoads,recvWait,recvData,recvPollLoads,post,elems
    p_ll_detail = re.compile(r"NCCL_PROFILE_LL_DETAIL,(\d+),(-?\d+),([\w_]+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)")
    # NCCL_PROFILE_LL128_DETAIL,channel,chunk,op,loadBegin,loadFinish,dstStore,sendStore,waitSendCredit,waitSendFifo,waitSendPolls,barrier,syncWarp,recvWait,recvData,recvPollLoads,post,elems
    p_ll128_detail = re.compile(r"NCCL_PROFILE_LL128_DETAIL,(\d+),(-?\d+),([\w_]+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)")

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

            m_ll = p_ll.search(line)
            if m_ll:
                ll_data.append({
                    'Channel': int(m_ll.group(1)),
                    'ChunkNum': int(m_ll.group(2)),
                    'Operation': m_ll.group(3),
                    'Load_Cycles': int(m_ll.group(4)),
                    'Send_Cycles': int(m_ll.group(5)),
                    'Sync_Cycles': int(m_ll.group(6)),
                    'Total_Cycles': int(m_ll.group(7)),
                    'Elems': int(m_ll.group(8))
                })
                continue

            m_ll128 = p_ll128.search(line)
            if m_ll128:
                ll128_data.append({
                    'Channel': int(m_ll128.group(1)),
                    'ChunkNum': int(m_ll128.group(2)),
                    'Operation': m_ll128.group(3),
                    'Load_Cycles': int(m_ll128.group(4)),
                    'Send_Cycles': int(m_ll128.group(5)),
                    'Sync_Cycles': int(m_ll128.group(6)),
                    'Total_Cycles': int(m_ll128.group(7)),
                    'Elems': int(m_ll128.group(8))
                })
                continue

            m_ll_detail = p_ll_detail.search(line)
            if m_ll_detail:
                ll_detail_data.append({
                    'Channel': int(m_ll_detail.group(1)),
                    'ChunkNum': int(m_ll_detail.group(2)),
                    'Operation': m_ll_detail.group(3),
                    'LoadBegin_Cycles': int(m_ll_detail.group(4)),
                    'LoadFinish_Cycles': int(m_ll_detail.group(5)),
                    'DstStore_Cycles': int(m_ll_detail.group(6)),
                    'SendStore_Cycles': int(m_ll_detail.group(7)),
                    'WaitSendCredit_Cycles': int(m_ll_detail.group(8)),
                    'WaitSendFifo_Cycles': int(m_ll_detail.group(9)),
                    'WaitSendBarrier_Cycles': int(m_ll_detail.group(10)),
                    'WaitSendPolls': int(m_ll_detail.group(11)),
                    'RecvBegin_Cycles': int(m_ll_detail.group(12)),
                    'RecvBeginLoads': int(m_ll_detail.group(13)),
                    'RecvWait_Cycles': int(m_ll_detail.group(14)),
                    'RecvData_Cycles': int(m_ll_detail.group(15)),
                    'RecvPollLoads': int(m_ll_detail.group(16)),
                    'Post_Cycles': int(m_ll_detail.group(17)),
                    'Elems': int(m_ll_detail.group(18))
                })
                continue

            m_ll128_detail = p_ll128_detail.search(line)
            if m_ll128_detail:
                ll128_detail_data.append({
                    'Channel': int(m_ll128_detail.group(1)),
                    'ChunkNum': int(m_ll128_detail.group(2)),
                    'Operation': m_ll128_detail.group(3),
                    'LoadBegin_Cycles': int(m_ll128_detail.group(4)),
                    'LoadFinish_Cycles': int(m_ll128_detail.group(5)),
                    'DstStore_Cycles': int(m_ll128_detail.group(6)),
                    'SendStore_Cycles': int(m_ll128_detail.group(7)),
                    'WaitSendCredit_Cycles': int(m_ll128_detail.group(8)),
                    'WaitSendFifo_Cycles': int(m_ll128_detail.group(9)),
                    'WaitSendPolls': int(m_ll128_detail.group(10)),
                    'Barrier_Cycles': int(m_ll128_detail.group(11)),
                    'SyncWarp_Cycles': int(m_ll128_detail.group(12)),
                    'RecvWait_Cycles': int(m_ll128_detail.group(13)),
                    'RecvData_Cycles': int(m_ll128_detail.group(14)),
                    'RecvPollLoads': int(m_ll128_detail.group(15)),
                    'Post_Cycles': int(m_ll128_detail.group(16)),
                    'Elems': int(m_ll128_detail.group(17))
                })
                continue
                
    return (
        pd.DataFrame(chunk_data),
        pd.DataFrame(slice_data),
        pd.DataFrame(tile_data),
        pd.DataFrame(ll_data),
        pd.DataFrame(ll128_data),
        pd.DataFrame(ll_detail_data),
        pd.DataFrame(ll128_detail_data),
    )

def main():
    parser = argparse.ArgumentParser(description="Process NCCL Profile Logs (including TILE)")
    parser.add_argument("logfile", help="Path to log file")
    parser.add_argument("output", help="Optional output filename prefix", nargs='?')
    parser.add_argument("--freq", type=float, help="GPU frequency in GHz for time conversion (e.g., 1.98)", default=1.0)
    
    args = parser.parse_args()

    print(f"Parsing {args.logfile}...")
    df_chunk, df_slice, df_tile, df_ll, df_ll128, df_ll_detail, df_ll128_detail = parse_log(args.logfile)
    meta = parse_header(args.logfile)

    if df_chunk.empty and df_slice.empty and df_tile.empty and df_ll.empty and df_ll128.empty and df_ll_detail.empty and df_ll128_detail.empty:
        print("No profile data found in log.")
        sys.exit(1)

    print(f"Converting cycles to microseconds using {args.freq} GHz")
    time_col = 'Time_us'
    
    for df in [df_chunk, df_slice, df_tile]:
        if not df.empty:
            df['Time_us'] = df['Time_Cycles'] / (args.freq * 1000.0)
    if not df_ll.empty:
        for c in ['Load_Cycles', 'Send_Cycles', 'Sync_Cycles', 'Total_Cycles']:
            df_ll[c.replace('_Cycles', '_us')] = df_ll[c] / (args.freq * 1000.0)
    if not df_ll128.empty:
        for c in ['Load_Cycles', 'Send_Cycles', 'Sync_Cycles', 'Total_Cycles']:
            df_ll128[c.replace('_Cycles', '_us')] = df_ll128[c] / (args.freq * 1000.0)
    if not df_ll_detail.empty:
        for c in ['LoadBegin_Cycles', 'LoadFinish_Cycles', 'DstStore_Cycles', 'SendStore_Cycles', 'WaitSendCredit_Cycles',
                  'WaitSendFifo_Cycles', 'WaitSendBarrier_Cycles', 'RecvBegin_Cycles', 'RecvWait_Cycles', 'RecvData_Cycles', 'Post_Cycles']:
            df_ll_detail[c.replace('_Cycles', '_us')] = df_ll_detail[c] / (args.freq * 1000.0)
    if not df_ll128_detail.empty:
        for c in ['LoadBegin_Cycles', 'LoadFinish_Cycles', 'DstStore_Cycles', 'SendStore_Cycles', 'WaitSendCredit_Cycles',
                  'WaitSendFifo_Cycles', 'Barrier_Cycles', 'SyncWarp_Cycles', 'RecvWait_Cycles', 'RecvData_Cycles', 'Post_Cycles']:
            df_ll128_detail[c.replace('_Cycles', '_us')] = df_ll128_detail[c] / (args.freq * 1000.0)

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

    # 4. LL Primitive Data (load / send / sync breakdown)
    if not df_ll.empty:
        df_ll = df_ll.sort_values(by=['ChunkNum', 'Channel', 'Operation'])
        dfs_to_save['LL_Raw'] = df_ll

        ll_summary_ch = df_ll.groupby(['Channel', 'ChunkNum', 'Operation']).agg(
            Load_us_mean=('Load_us', 'mean'),
            Load_us_max=('Load_us', 'max'),
            Send_us_mean=('Send_us', 'mean'),
            Send_us_max=('Send_us', 'max'),
            Sync_us_mean=('Sync_us', 'mean'),
            Sync_us_max=('Sync_us', 'max'),
            Total_us_mean=('Total_us', 'mean'),
            Total_us_max=('Total_us', 'max'),
            Elems_mean=('Elems', 'mean'),
            Count=('Operation', 'count')
        ).reset_index()
        dfs_to_save['LL_Summary_Channel'] = ll_summary_ch

        ll_summary_global = df_ll.groupby(['Operation']).agg(
            Load_us_mean=('Load_us', 'mean'),
            Load_us_max=('Load_us', 'max'),
            Send_us_mean=('Send_us', 'mean'),
            Send_us_max=('Send_us', 'max'),
            Sync_us_mean=('Sync_us', 'mean'),
            Sync_us_max=('Sync_us', 'max'),
            Total_us_mean=('Total_us', 'mean'),
            Total_us_max=('Total_us', 'max'),
            Elems_mean=('Elems', 'mean'),
            Count=('Operation', 'count')
        ).reset_index()
        dfs_to_save['LL_Summary_Global'] = ll_summary_global

    # 5. LL128 Primitive Data (load / send / sync breakdown)
    if not df_ll128.empty:
        df_ll128 = df_ll128.sort_values(by=['ChunkNum', 'Channel', 'Operation'])
        dfs_to_save['LL128_Raw'] = df_ll128

        ll128_summary_ch = df_ll128.groupby(['Channel', 'ChunkNum', 'Operation']).agg(
            Load_us_mean=('Load_us', 'mean'),
            Load_us_max=('Load_us', 'max'),
            Send_us_mean=('Send_us', 'mean'),
            Send_us_max=('Send_us', 'max'),
            Sync_us_mean=('Sync_us', 'mean'),
            Sync_us_max=('Sync_us', 'max'),
            Total_us_mean=('Total_us', 'mean'),
            Total_us_max=('Total_us', 'max'),
            Elems_mean=('Elems', 'mean'),
            Count=('Operation', 'count')
        ).reset_index()
        dfs_to_save['LL128_Summary_Channel'] = ll128_summary_ch

        ll128_summary_global = df_ll128.groupby(['Operation']).agg(
            Load_us_mean=('Load_us', 'mean'),
            Load_us_max=('Load_us', 'max'),
            Send_us_mean=('Send_us', 'mean'),
            Send_us_max=('Send_us', 'max'),
            Sync_us_mean=('Sync_us', 'mean'),
            Sync_us_max=('Sync_us', 'max'),
            Total_us_mean=('Total_us', 'mean'),
            Total_us_max=('Total_us', 'max'),
            Elems_mean=('Elems', 'mean'),
            Count=('Operation', 'count')
        ).reset_index()
        dfs_to_save['LL128_Summary_Global'] = ll128_summary_global

    # 6. LL Detail Data
    if not df_ll_detail.empty:
        df_ll_detail = df_ll_detail.sort_values(by=['ChunkNum', 'Channel', 'Operation'])
        dfs_to_save['LL_Detail_Raw'] = df_ll_detail
        ll_detail_summary = df_ll_detail.groupby(['Operation']).agg(
            LoadBegin_us_mean=('LoadBegin_us', 'mean'),
            LoadFinish_us_mean=('LoadFinish_us', 'mean'),
            DstStore_us_mean=('DstStore_us', 'mean'),
            SendStore_us_mean=('SendStore_us', 'mean'),
            WaitSendCredit_us_mean=('WaitSendCredit_us', 'mean'),
            WaitSendFifo_us_mean=('WaitSendFifo_us', 'mean'),
            WaitSendBarrier_us_mean=('WaitSendBarrier_us', 'mean'),
            RecvBegin_us_mean=('RecvBegin_us', 'mean'),
            RecvWait_us_mean=('RecvWait_us', 'mean'),
            RecvData_us_mean=('RecvData_us', 'mean'),
            Post_us_mean=('Post_us', 'mean'),
            WaitSendPolls_mean=('WaitSendPolls', 'mean'),
            RecvBeginLoads_mean=('RecvBeginLoads', 'mean'),
            RecvPollLoads_mean=('RecvPollLoads', 'mean'),
            Elems_mean=('Elems', 'mean'),
            Count=('Operation', 'count')
        ).reset_index()
        dfs_to_save['LL_Detail_Summary'] = ll_detail_summary

    # 7. LL128 Detail Data
    if not df_ll128_detail.empty:
        df_ll128_detail = df_ll128_detail.sort_values(by=['ChunkNum', 'Channel', 'Operation'])
        dfs_to_save['LL128_Detail_Raw'] = df_ll128_detail
        ll128_detail_summary = df_ll128_detail.groupby(['Operation']).agg(
            LoadBegin_us_mean=('LoadBegin_us', 'mean'),
            LoadFinish_us_mean=('LoadFinish_us', 'mean'),
            DstStore_us_mean=('DstStore_us', 'mean'),
            SendStore_us_mean=('SendStore_us', 'mean'),
            WaitSendCredit_us_mean=('WaitSendCredit_us', 'mean'),
            WaitSendFifo_us_mean=('WaitSendFifo_us', 'mean'),
            Barrier_us_mean=('Barrier_us', 'mean'),
            SyncWarp_us_mean=('SyncWarp_us', 'mean'),
            RecvWait_us_mean=('RecvWait_us', 'mean'),
            RecvData_us_mean=('RecvData_us', 'mean'),
            Post_us_mean=('Post_us', 'mean'),
            WaitSendPolls_mean=('WaitSendPolls', 'mean'),
            RecvPollLoads_mean=('RecvPollLoads', 'mean'),
            Elems_mean=('Elems', 'mean'),
            Count=('Operation', 'count')
        ).reset_index()
        dfs_to_save['LL128_Detail_Summary'] = ll128_detail_summary


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
