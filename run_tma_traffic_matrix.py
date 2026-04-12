#!/usr/bin/env python3
import argparse
import csv
import os
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path


RESULT_RE = re.compile(
    r"\[(TMA|SIMPLE)\]\s+"
    r"avg=([0-9.]+)\s+us\s+"
    r"min=([0-9.]+)\s+us\s+"
    r"p50=([0-9.]+)\s+us\s+"
    r"p95=([0-9.]+)\s+us\s+"
    r"max=([0-9.]+)\s+us\s+"
    r"std=([0-9.]+)\s+us\s+"
    r"BW=([0-9.]+)\s+GB/s\s+"
    r"wrong=(\d+)"
)

TILE_RE = re.compile(r"NCCL_PROFILE_TILE,(\d+),(\d+),(\d+),(\d+),([\w_]+),(\d+)")


def parse_csv_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def percentile(values, q):
    if not values:
        return ""
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def run_cmd(cmd, cwd, log_path=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if log_path is not None:
        log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {' '.join(cmd)}\n"
            f"log: {log_path if log_path else '<not saved>'}"
        )
    return proc.stdout


def compile_bench(args, root):
    if args.no_compile:
        return
    cmd = [
        args.nvcc_bin,
        f"-arch={args.arch}",
        "-O3",
        "-std=c++20",
        str(args.source),
        "-o",
        str(args.binary),
    ]
    if args.tile_profile:
        cmd.insert(4, "-DENABLE_SLICE_PROFILING=0")
        cmd.insert(4, "-DENABLE_TILE_PROFILING=1")
        cmd.insert(4, "-DENABLE_PROFILING=0")
    else:
        cmd.insert(4, "-DENABLE_PROFILING=0")
    if args.extra_nvcc_flags:
        cmd.extend(args.extra_nvcc_flags.split())
    run_cmd(cmd, root, args.output_dir / "compile.log")


def parse_result(stdout, expected_proto):
    rows = []
    for line in stdout.splitlines():
        match = RESULT_RE.search(line)
        if not match:
            continue
        proto = match.group(1)
        if expected_proto and proto != expected_proto:
            continue
        rows.append(
            {
                "proto_name": proto,
                "avg_us": float(match.group(2)),
                "min_us": float(match.group(3)),
                "p50_us": float(match.group(4)),
                "p95_us": float(match.group(5)),
                "max_us": float(match.group(6)),
                "std_us": float(match.group(7)),
                "bw_gbps": float(match.group(8)),
                "wrong": int(match.group(9)),
            }
        )
    return rows


def parse_tile_events(stdout, freq_ghz):
    rows = []
    for line in stdout.splitlines():
        match = TILE_RE.search(line)
        if not match:
            continue
        cycles = int(match.group(6))
        row = {
            "channel": int(match.group(1)),
            "chunk": int(match.group(2)),
            "slice": int(match.group(3)),
            "tile": int(match.group(4)),
            "block_type": match.group(5),
            "time_cycles": cycles,
        }
        if freq_ghz > 0:
            row["time_us"] = cycles / (freq_ghz * 1000.0)
        rows.append(row)
    return rows


def mean(values):
    return statistics.fmean(values) if values else ""


def stdev(values):
    return statistics.stdev(values) if len(values) >= 2 else 0.0 if values else ""


def summarize(rows):
    groups = {}
    for row in rows:
        key = (
            row["proto_name"],
            row["traffic_mode"],
            row["traffic_target"],
            row["traffic_mb"],
            row["traffic_grid"],
            row["traffic_block"],
            row["total_mb"],
            row["grid"],
            row["block"],
            row["pipe"],
            row["smem_kb"],
            row["tma_slice_kb"],
            row["tma_tile_kb"],
        )
        groups.setdefault(key, []).append(row)

    out = []
    for key, group in sorted(groups.items()):
        avg_us = [r["avg_us"] for r in group]
        p95_us = [r["p95_us"] for r in group]
        max_us = [r["max_us"] for r in group]
        std_us = [r["std_us"] for r in group]
        bw_gbps = [r["bw_gbps"] for r in group]
        out.append(
            {
                "proto_name": key[0],
                "traffic_mode": key[1],
                "traffic_target": key[2],
                "traffic_mb": key[3],
                "traffic_grid": key[4],
                "traffic_block": key[5],
                "total_mb": key[6],
                "grid": key[7],
                "block": key[8],
                "pipe": key[9],
                "smem_kb": key[10],
                "tma_slice_kb": key[11],
                "tma_tile_kb": key[12],
                "runs": len(group),
                "wrong_max": max(r["wrong"] for r in group),
                "avg_us_mean": mean(avg_us),
                "avg_us_stdev": stdev(avg_us),
                "p95_us_mean": mean(p95_us),
                "max_us_mean": mean(max_us),
                "std_us_mean": mean(std_us),
                "bw_gbps_mean": mean(bw_gbps),
                "bw_gbps_stdev": stdev(bw_gbps),
            }
        )
    return out


def summarize_tile_events(rows):
    groups = {}
    for row in rows:
        key = (
            row["block_type"],
            row["traffic_mode"],
            row["traffic_target"],
            row["traffic_mb"],
            row["traffic_grid"],
            row["traffic_block"],
            row["total_mb"],
            row["grid"],
            row["block"],
            row["pipe"],
            row["smem_kb"],
            row["tma_slice_kb"],
            row["tma_tile_kb"],
        )
        groups.setdefault(key, []).append(row)

    out = []
    for key, group in sorted(groups.items()):
        cycles = [r["time_cycles"] for r in group]
        time_us = [r["time_us"] for r in group if "time_us" in r]
        row = {
            "block_type": key[0],
            "traffic_mode": key[1],
            "traffic_target": key[2],
            "traffic_mb": key[3],
            "traffic_grid": key[4],
            "traffic_block": key[5],
            "total_mb": key[6],
            "grid": key[7],
            "block": key[8],
            "pipe": key[9],
            "smem_kb": key[10],
            "tma_slice_kb": key[11],
            "tma_tile_kb": key[12],
            "samples": len(group),
            "time_cycles_mean": mean(cycles),
            "time_cycles_stdev": stdev(cycles),
            "time_cycles_min": min(cycles),
            "time_cycles_p50": percentile(cycles, 0.50),
            "time_cycles_p95": percentile(cycles, 0.95),
            "time_cycles_p99": percentile(cycles, 0.99),
            "time_cycles_max": max(cycles),
        }
        if time_us:
            row.update(
                {
                    "time_us_mean": mean(time_us),
                    "time_us_stdev": stdev(time_us),
                    "time_us_min": min(time_us),
                    "time_us_p50": percentile(time_us, 0.50),
                    "time_us_p95": percentile(time_us, 0.95),
                    "time_us_p99": percentile(time_us, 0.99),
                    "time_us_max": max(time_us),
                }
            )
        out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(path, raw_rows, summary_rows, tile_rows, tile_summary_rows):
    try:
        import pandas as pd
    except ImportError:
        return False

    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame(raw_rows).to_excel(writer, sheet_name="Raw", index=False)
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            if tile_rows:
                pd.DataFrame(tile_rows).to_excel(writer, sheet_name="TileRaw", index=False)
            if tile_summary_rows:
                pd.DataFrame(tile_summary_rows).to_excel(writer, sheet_name="TileSummary", index=False)
    except ImportError:
        return False
    return True


def main():
    root = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="Run simple_tma_transfer_bench traffic contention matrix and write Excel-ready results."
    )
    parser.add_argument("--source", type=Path, default=root / "simple_tma_transfer_bench.cu")
    parser.add_argument("--binary", type=Path, default=root / "simple_tma_transfer_bench")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / f"tma_traffic_{stamp}")
    parser.add_argument("--nvcc-bin", default=os.environ.get("NVCC_BIN", "nvcc"))
    parser.add_argument("--arch", default=os.environ.get("ARCH", "sm_90a"))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--extra-nvcc-flags", default="")
    parser.add_argument("--tile-profile", action="store_true", help="Enable tile profiling and parse all NCCL_PROFILE_TILE rows.")
    parser.add_argument("--freq-ghz", type=float, default=1.0, help="clock64 frequency for cycles->us conversion.")

    parser.add_argument("--total-mb", type=int, default=128)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--block", type=int, default=544)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--src", type=int, default=0)
    parser.add_argument("--dst", type=int, default=1)
    parser.add_argument("--proto", type=int, default=1, help="1=TMA, 2=SIMPLE, 0=both")
    parser.add_argument("--pipe", type=int, default=2)
    parser.add_argument("--smem-kb", type=int, default=64)
    parser.add_argument("--simple-slice-kb", type=int, default=1024)
    parser.add_argument("--tma-slice-kb", type=int, default=1024)
    parser.add_argument("--tma-tile-kb", type=int, default=32)
    parser.add_argument("--verify", type=int, default=0)

    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--traffic-modes", default="none,read,write,readwrite")
    parser.add_argument("--traffic-targets", default="src,dst,both")
    parser.add_argument("--traffic-mb", type=int, default=1024)
    parser.add_argument("--traffic-grid", type=int, default=16)
    parser.add_argument("--traffic-block", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compile_bench(args, root)

    raw_rows = []
    tile_rows = []
    modes = parse_csv_list(args.traffic_modes)
    targets = parse_csv_list(args.traffic_targets)
    proto_name = {0: None, 1: "TMA", 2: "SIMPLE"}[args.proto]

    for mode in modes:
        target_list = ["src"] if mode == "none" else targets
        for target in target_list:
            target_label = "none" if mode == "none" else target
            for rep in range(1, args.repeats + 1):
                log_name = f"mode-{mode}_target-{target_label}_rep-{rep}.log"
                log_path = args.output_dir / log_name
                cmd = [
                    str(args.binary),
                    "--total_mb", str(args.total_mb),
                    "--grid", str(args.grid),
                    "--block", str(args.block),
                    "--iters", str(args.iters),
                    "--warmup", str(args.warmup),
                    "--src", str(args.src),
                    "--dst", str(args.dst),
                    "--proto", str(args.proto),
                    "--pipe", str(args.pipe),
                    "--smem_kb", str(args.smem_kb),
                    "--simple_slice_kb", str(args.simple_slice_kb),
                    "--tma_slice_kb", str(args.tma_slice_kb),
                    "--tma_tile_kb", str(args.tma_tile_kb),
                    "--verify", str(args.verify),
                    "--traffic_mode", mode,
                    "--traffic_target", target,
                    "--traffic_mb", str(args.traffic_mb),
                    "--traffic_grid", str(args.traffic_grid),
                    "--traffic_block", str(args.traffic_block),
                ]
                print(f"[run] mode={mode} target={target_label} rep={rep}")
                stdout = run_cmd(cmd, root, log_path)
                parsed = parse_result(stdout, proto_name)
                if not parsed:
                    raise RuntimeError(f"no result line parsed from {log_path}")
                parsed_tiles = parse_tile_events(stdout, args.freq_ghz) if args.tile_profile else []
                for row in parsed:
                    row.update(
                        {
                            "repeat": rep,
                            "traffic_mode": mode,
                            "traffic_target": target_label,
                            "traffic_mb": args.traffic_mb,
                            "traffic_grid": args.traffic_grid,
                            "traffic_block": args.traffic_block,
                            "total_mb": args.total_mb,
                            "grid": args.grid,
                            "block": args.block,
                            "pipe": args.pipe,
                            "smem_kb": args.smem_kb,
                            "simple_slice_kb": args.simple_slice_kb,
                            "tma_slice_kb": args.tma_slice_kb,
                            "tma_tile_kb": args.tma_tile_kb,
                            "log": str(log_path),
                        }
                    )
                    raw_rows.append(row)
                for row in parsed_tiles:
                    row.update(
                        {
                            "repeat": rep,
                            "traffic_mode": mode,
                            "traffic_target": target_label,
                            "traffic_mb": args.traffic_mb,
                            "traffic_grid": args.traffic_grid,
                            "traffic_block": args.traffic_block,
                            "total_mb": args.total_mb,
                            "grid": args.grid,
                            "block": args.block,
                            "pipe": args.pipe,
                            "smem_kb": args.smem_kb,
                            "tma_slice_kb": args.tma_slice_kb,
                            "tma_tile_kb": args.tma_tile_kb,
                            "log": str(log_path),
                        }
                    )
                    tile_rows.append(row)

    summary_rows = summarize(raw_rows)
    tile_summary_rows = summarize_tile_events(tile_rows) if tile_rows else []
    raw_csv = args.output_dir / "raw_results.csv"
    summary_csv = args.output_dir / "summary.csv"
    tile_csv = args.output_dir / "tile_raw.csv"
    tile_summary_csv = args.output_dir / "tile_summary.csv"
    xlsx = args.output_dir / "tma_traffic_results.xlsx"
    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(tile_csv, tile_rows)
    write_csv(tile_summary_csv, tile_summary_rows)
    wrote_xlsx = write_xlsx(xlsx, raw_rows, summary_rows, tile_rows, tile_summary_rows)

    print(f"[done] raw csv: {raw_csv}")
    print(f"[done] summary csv: {summary_csv}")
    if tile_rows:
        print(f"[done] tile raw csv: {tile_csv}")
        print(f"[done] tile summary csv: {tile_summary_csv}")
    if wrote_xlsx:
        print(f"[done] xlsx: {xlsx}")
    else:
        print("[info] pandas/openpyxl not available; use the CSV files in Excel.")


if __name__ == "__main__":
    main()
