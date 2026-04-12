#!/usr/bin/env python3
import argparse
import csv
import os
import re
import selectors
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


def parse_str_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value):
    return [int(x) for x in parse_str_list(value)]


def percentile(values, q):
    if not values:
        return ""
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def mean(values):
    return statistics.fmean(values) if values else ""


def stdev(values):
    return statistics.stdev(values) if len(values) >= 2 else 0.0 if values else ""


def run_cmd(cmd, cwd, log_path=None, timeout_sec=0):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    output = []
    start = datetime.now()

    try:
        with (log_path.open("w") if log_path else open(os.devnull, "w")) as logf:
            while True:
                for key, _ in selector.select(timeout=0.2):
                    line = key.fileobj.readline()
                    if line:
                        output.append(line)
                        logf.write(line)
                        logf.flush()

                rc = proc.poll()
                if rc is not None:
                    rest = proc.stdout.read()
                    if rest:
                        output.append(rest)
                        logf.write(rest)
                    break

                if timeout_sec > 0 and (datetime.now() - start).total_seconds() > timeout_sec:
                    proc.kill()
                    proc.wait()
                    tail = "".join(output[-40:])
                    raise TimeoutError(
                        f"command timed out after {timeout_sec}s: {' '.join(cmd)}\n"
                        f"log: {log_path if log_path else '<not saved>'}\n"
                        f"last output lines:\n{tail}"
                    )
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise

    if proc.returncode != 0:
        tail = "".join(output[-40:])
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {' '.join(cmd)}\n"
            f"log: {log_path if log_path else '<not saved>'}\n"
            f"last output lines:\n{tail}"
        )
    return "".join(output)


def compile_bench(args, root):
    if args.no_compile:
        return

    cmd = [
        args.nvcc_bin,
        f"-arch={args.arch}",
        "-O3",
        "-std=c++20",
        "-DENABLE_PROFILING=0",
        "-DENABLE_TILE_PROFILING=1",
        "-DENABLE_SLICE_PROFILING=0",
        str(args.source),
        "-o",
        str(args.binary),
    ]
    if args.extra_nvcc_flags:
        cmd.extend(args.extra_nvcc_flags.split())
    print("[compile] " + " ".join(cmd))
    run_cmd(cmd, root, args.output_dir / "compile.log")


def parse_bench_result(stdout):
    for line in stdout.splitlines():
        m = RESULT_RE.search(line)
        if m and m.group(1) == "TMA":
            return {
                "tma_avg_us": float(m.group(2)),
                "tma_p95_us": float(m.group(5)),
                "tma_max_us": float(m.group(6)),
                "tma_std_us": float(m.group(7)),
                "tma_bw_gbps": float(m.group(8)),
                "wrong": int(m.group(9)),
            }
    return {}


def parse_send_wait(stdout, freq_ghz):
    rows = []
    for line in stdout.splitlines():
        m = TILE_RE.search(line)
        if not m or m.group(5) != "SEND_WAIT":
            continue
        cycles = int(m.group(6))
        row = {
            "channel": int(m.group(1)),
            "chunk": int(m.group(2)),
            "slice": int(m.group(3)),
            "tile": int(m.group(4)),
            "time_cycles": cycles,
        }
        if freq_ghz > 0:
            row["time_us"] = cycles / (freq_ghz * 1000.0)
        rows.append(row)
    return rows


def summarize_group(rows):
    cycles = [r["time_cycles"] for r in rows]
    out = {
        "samples": len(cycles),
        "time_cycles_mean": mean(cycles),
        "time_cycles_p50": percentile(cycles, 0.50),
        "time_cycles_p95": percentile(cycles, 0.95),
        "time_cycles_p99": percentile(cycles, 0.99),
        "time_cycles_max": max(cycles) if cycles else "",
    }
    us = [r["time_us"] for r in rows if "time_us" in r]
    if us:
        out.update(
            {
                "time_us_mean": mean(us),
                "time_us_p50": percentile(us, 0.50),
                "time_us_p95": percentile(us, 0.95),
                "time_us_p99": percentile(us, 0.99),
                "time_us_max": max(us),
            }
        )
    return out


def summarize(raw_rows):
    groups = {}
    for row in raw_rows:
        key = (
            row["traffic_mode"],
            row["traffic_target"],
            row["traffic_grid"],
            row["traffic_block"],
            row["traffic_rounds"],
            row["traffic_mb"],
            row["total_mb"],
            row["grid"],
            row["block"],
            row["pipe"],
            row["smem_kb"],
            row["tma_slice_kb"],
            row["tma_tile_kb"],
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for key, rows in sorted(groups.items()):
        out = {
            "traffic_mode": key[0],
            "traffic_target": key[1],
            "traffic_grid": key[2],
            "traffic_block": key[3],
            "traffic_rounds": key[4],
            "traffic_mb": key[5],
            "total_mb": key[6],
            "grid": key[7],
            "block": key[8],
            "pipe": key[9],
            "smem_kb": key[10],
            "tma_slice_kb": key[11],
            "tma_tile_kb": key[12],
        }
        out.update(summarize_group(rows))
        out["runs"] = len({r["run_id"] for r in rows})
        out["wrong_max"] = max(r.get("wrong", 0) for r in rows)
        out["tma_avg_us_mean"] = mean([r["tma_avg_us"] for r in rows if "tma_avg_us" in r])
        out["tma_bw_gbps_mean"] = mean([r["tma_bw_gbps"] for r in rows if "tma_bw_gbps" in r])
        summary.append(out)
    return summary


def add_baseline_deltas(summary_rows):
    baselines = [r for r in summary_rows if r["traffic_mode"] == "none"]
    if not baselines:
        return summary_rows

    baseline = baselines[0]
    metrics = [
        "time_cycles_mean",
        "time_cycles_p50",
        "time_cycles_p95",
        "time_cycles_p99",
        "time_cycles_max",
    ]
    for row in summary_rows:
        for metric in metrics:
            base = baseline.get(metric, "")
            val = row.get(metric, "")
            if base == "" or val == "":
                continue
            row[f"{metric}_delta_vs_base"] = val - base
            row[f"{metric}_ratio_vs_base"] = val / base if base else ""
    return summary_rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(path, raw_rows, summary_rows):
    try:
        import pandas as pd
    except ImportError:
        return False

    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="SendWaitSummary", index=False)
            pd.DataFrame(raw_rows).to_excel(writer, sheet_name="SendWaitRaw", index=False)
    except ImportError:
        return False
    return True


def safe_name(value):
    return str(value).replace(",", "-").replace("/", "_").replace(" ", "")


def default_output_dir(root, args, stamp):
    name = (
        f"tma_wait_"
        f"mb{args.total_mb}_g{args.grid}_b{args.block}_"
        f"p{args.pipe}_sm{args.smem_kb}_"
        f"slice{args.tma_slice_kb}_tile{args.tma_tile_kb}_"
        f"m{safe_name(args.traffic_modes)}_"
        f"tg{safe_name(args.traffic_grids)}_"
        f"tb{safe_name(args.traffic_blocks)}_"
        f"tr{safe_name(args.traffic_rounds)}_"
        f"rep{args.repeats}_{stamp}"
    )
    return root / "results" / name


def make_run_configs(args):
    modes = parse_str_list(args.traffic_modes)
    targets = parse_str_list(args.traffic_targets)
    traffic_grids = parse_int_list(args.traffic_grids)
    traffic_blocks = parse_int_list(args.traffic_blocks)
    traffic_rounds = parse_int_list(args.traffic_rounds)

    configs = []
    if "none" in modes:
        for rep in range(1, args.repeats + 1):
            configs.append(("none", "none", 0, 0, 0, rep))

    for mode in modes:
        if mode == "none":
            continue
        for target in targets:
            for traffic_grid in traffic_grids:
                for traffic_block in traffic_blocks:
                    for rounds in traffic_rounds:
                        for rep in range(1, args.repeats + 1):
                            configs.append((mode, target, traffic_grid, traffic_block, rounds, rep))
    return configs


def main():
    root = Path(__file__).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="Sweep traffic parameters and summarize TMA SEND_WAIT tile timing."
    )
    parser.add_argument("--source", type=Path, default=root / "simple_tma_transfer_bench.cu")
    parser.add_argument("--binary", type=Path, default=root / "simple_tma_transfer_bench")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--nvcc-bin", default=os.environ.get("NVCC_BIN", "nvcc"))
    parser.add_argument("--arch", default=os.environ.get("ARCH", "sm_90a"))
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--extra-nvcc-flags", default="")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--freq-ghz", type=float, default=1.0)

    parser.add_argument("--total-mb", type=int, default=128)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--block", type=int, default=544)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--src", type=int, default=0)
    parser.add_argument("--dst", type=int, default=1)
    parser.add_argument("--pipe", type=int, default=2)
    parser.add_argument("--smem-kb", type=int, default=64)
    parser.add_argument("--tma-slice-kb", type=int, default=1024)
    parser.add_argument("--tma-tile-kb", type=int, default=32)
    parser.add_argument("--traffic-mb", type=int, default=256)

    parser.add_argument("--traffic-modes", default="none,read,write,readwrite")
    parser.add_argument("--traffic-targets", default="src,dst,both")
    parser.add_argument("--traffic-grids", default="1,2,4,8")
    parser.add_argument("--traffic-blocks", default="256")
    parser.add_argument("--traffic-rounds", default="64,256,1024")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir(root, args, stamp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compile_bench(args, root)

    raw_rows = []
    configs = make_run_configs(args)
    for run_idx, (mode, target, traffic_grid, traffic_block, rounds, rep) in enumerate(configs, 1):
        target_arg = "src" if mode == "none" else target
        traffic_grid_arg = 1 if mode == "none" else traffic_grid
        traffic_block_arg = 256 if mode == "none" else traffic_block
        rounds_arg = 1 if mode == "none" else rounds

        run_id = (
            f"{run_idx:04d}_mode-{mode}_target-{target}_tg-{traffic_grid}_"
            f"tb-{traffic_block}_tr-{rounds}_rep-{rep}"
        )
        log_path = args.output_dir / f"{run_id}.log"
        cmd = [
            str(args.binary),
            "--total_mb", str(args.total_mb),
            "--grid", str(args.grid),
            "--block", str(args.block),
            "--iters", str(args.iters),
            "--warmup", str(args.warmup),
            "--src", str(args.src),
            "--dst", str(args.dst),
            "--proto", "1",
            "--pipe", str(args.pipe),
            "--smem_kb", str(args.smem_kb),
            "--tma_slice_kb", str(args.tma_slice_kb),
            "--tma_tile_kb", str(args.tma_tile_kb),
            "--verify", "0",
            "--traffic_mode", mode,
            "--traffic_target", target_arg,
            "--traffic_mb", str(args.traffic_mb),
            "--traffic_grid", str(traffic_grid_arg),
            "--traffic_block", str(traffic_block_arg),
            "--traffic_rounds", str(rounds_arg),
        ]
        print(
            f"[run {run_idx}/{len(configs)}] mode={mode} target={target} "
            f"traffic_grid={traffic_grid} traffic_block={traffic_block} "
            f"traffic_rounds={rounds} rep={rep}"
        )
        stdout = run_cmd(cmd, root, log_path, args.timeout_sec)
        bench = parse_bench_result(stdout)
        wait_rows = parse_send_wait(stdout, args.freq_ghz)
        if not wait_rows:
            raise RuntimeError(f"no SEND_WAIT tile rows parsed from {log_path}")

        for row in wait_rows:
            row.update(
                {
                    "run_id": run_id,
                    "repeat": rep,
                    "traffic_mode": mode,
                    "traffic_target": "none" if mode == "none" else target,
                    "traffic_grid": 0 if mode == "none" else traffic_grid,
                    "traffic_block": 0 if mode == "none" else traffic_block,
                    "traffic_rounds": 0 if mode == "none" else rounds,
                    "traffic_mb": args.traffic_mb,
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
            row.update(bench)
            raw_rows.append(row)

    summary_rows = add_baseline_deltas(summarize(raw_rows))

    raw_csv = args.output_dir / "send_wait_raw.csv"
    summary_csv = args.output_dir / "send_wait_summary.csv"
    xlsx = args.output_dir / "send_wait_sweep.xlsx"
    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summary_rows)
    wrote_xlsx = write_xlsx(xlsx, raw_rows, summary_rows)

    print(f"[done] raw: {raw_csv}")
    print(f"[done] summary: {summary_csv}")
    if wrote_xlsx:
        print(f"[done] xlsx: {xlsx}")
    else:
        print("[info] pandas/openpyxl unavailable; use CSV files in Excel.")


if __name__ == "__main__":
    main()
