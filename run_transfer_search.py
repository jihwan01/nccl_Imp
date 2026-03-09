#!/usr/bin/env python3
import argparse
import csv
import ctypes
import dataclasses
import datetime as dt
import itertools
import math
import os
import pathlib
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class BenchConfig:
    total_mb: int
    grid: int
    block: int
    smem_kb: int
    pipe: int
    simple_slice_kb: int
    tma_slice_kb: int
    tma_tile_kb: int

    @property
    def total_bytes(self) -> int:
        return self.total_mb * 1024 * 1024

    @property
    def smem_bytes(self) -> int:
        return self.smem_kb * 1024

    @property
    def chunk_bytes(self) -> int:
        return self.total_bytes // self.grid

    @property
    def simple_slice_bytes(self) -> int:
        return self.simple_slice_kb * 1024

    @property
    def tma_slice_bytes(self) -> int:
        return self.tma_slice_kb * 1024

    @property
    def tma_tile_bytes(self) -> int:
        return self.tma_tile_kb * 1024

    @property
    def tma_slot_bytes(self) -> int:
        return self.smem_bytes // self.pipe


def parse_int_list(spec: str, name: str) -> List[int]:
    """
    Supports:
      - "64,128,256"
      - "64:256:64" (inclusive)
      - mixed: "64,128:256:64,512"
    """
    out: List[int] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" in token:
            parts = token.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid range token for {name}: '{token}'")
            start = int(parts[0])
            end = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            if step <= 0:
                raise ValueError(f"Step must be > 0 for {name}: '{token}'")
            if end < start:
                raise ValueError(f"Range end < start for {name}: '{token}'")
            v = start
            while v <= end:
                out.append(v)
                v += step
        else:
            out.append(int(token))

    dedup = list(dict.fromkeys(out))
    if not dedup:
        raise ValueError(f"Empty list for {name}")
    return dedup


def validate_cfg(cfg: BenchConfig, pair_slices: bool) -> Optional[str]:
    if cfg.total_mb <= 0:
        return "total_mb must be > 0"
    if cfg.grid <= 0:
        return "grid must be > 0"
    if cfg.block <= 0:
        return "block must be > 0"
    if cfg.smem_kb <= 0:
        return "smem_kb must be > 0"
    if cfg.pipe <= 0:
        return "pipe must be > 0"

    if cfg.smem_bytes % cfg.pipe != 0:
        return "smem_bytes % pipe != 0"
    if cfg.tma_slot_bytes % 16 != 0:
        return "tma_slot_bytes (smem_bytes/pipe) % 16 != 0"
    if cfg.total_bytes % cfg.grid != 0:
        return "total_bytes % grid != 0"

    if cfg.chunk_bytes <= 0:
        return "chunk_bytes <= 0"
    if cfg.chunk_bytes % 16 != 0:
        return "chunk_bytes % 16 != 0"

    if cfg.simple_slice_bytes <= 0 or cfg.simple_slice_bytes % 16 != 0:
        return "simple_slice_bytes invalid (must be >0 and 16B aligned)"
    if cfg.tma_slice_bytes <= 0 or cfg.tma_slice_bytes % 16 != 0:
        return "tma_slice_bytes invalid (must be >0 and 16B aligned)"
    if cfg.tma_tile_bytes <= 0 or cfg.tma_tile_bytes % 16 != 0:
        return "tma_tile_bytes invalid (must be >0 and 16B aligned)"

    if cfg.tma_tile_bytes != cfg.tma_slot_bytes:
        return "require slot_bytes == tile_bytes, i.e. smem_bytes/pipe == tma_tile_bytes"

    if pair_slices and cfg.simple_slice_kb != cfg.tma_slice_kb:
        return "pair_slices enabled and simple_slice_kb != tma_slice_kb"

    return None


def generate_configs(args: argparse.Namespace) -> Tuple[List[BenchConfig], List[Dict[str, str]]]:
    valid: List[BenchConfig] = []
    skipped: List[Dict[str, str]] = []
    seen = set()

    if args.derive_smem:
        combo_iter = itertools.product(
            args.total_mb_list,
            args.grid_list,
            args.block_list,
            args.pipe_list,
            args.simple_slice_kb_list,
            args.tma_slice_kb_list,
            args.tma_tile_kb_list,
        )
    else:
        combo_iter = itertools.product(
            args.total_mb_list,
            args.grid_list,
            args.block_list,
            args.smem_kb_list,
            args.pipe_list,
            args.simple_slice_kb_list,
            args.tma_slice_kb_list,
            args.tma_tile_kb_list,
        )

    for combo in combo_iter:
        if args.derive_smem:
            total_mb, grid, block, pipe, simple_slice_kb, tma_slice_kb, tma_tile_kb = combo
            smem_kb = tma_tile_kb * pipe
        else:
            total_mb, grid, block, smem_kb, pipe, simple_slice_kb, tma_slice_kb, tma_tile_kb = combo

        cfg = BenchConfig(
            total_mb=total_mb,
            grid=grid,
            block=block,
            smem_kb=smem_kb,
            pipe=pipe,
            simple_slice_kb=simple_slice_kb,
            tma_slice_kb=tma_slice_kb,
            tma_tile_kb=tma_tile_kb,
        )

        key = (
            cfg.total_mb,
            cfg.grid,
            cfg.block,
            cfg.smem_kb,
            cfg.pipe,
            cfg.simple_slice_kb,
            cfg.tma_slice_kb,
            cfg.tma_tile_kb,
        )
        if key in seen:
            continue
        seen.add(key)

        reason = validate_cfg(cfg, args.pair_slices)
        if reason is None:
            valid.append(cfg)
        else:
            skipped.append(
                {
                    "total_mb": str(cfg.total_mb),
                    "grid": str(cfg.grid),
                    "block": str(cfg.block),
                    "smem_kb": str(cfg.smem_kb),
                    "pipe": str(cfg.pipe),
                    "simple_slice_kb": str(cfg.simple_slice_kb),
                    "tma_slice_kb": str(cfg.tma_slice_kb),
                    "tma_tile_kb": str(cfg.tma_tile_kb),
                    "reason": reason,
                }
            )

    return valid, skipped


def query_max_dynamic_smem_bytes(device: int) -> Optional[int]:
    # Runtime API enum: cudaDevAttrMaxSharedMemoryPerBlockOptin
    cuda_dev_attr_max_shared_optin = 97
    # Fallback attr: cudaDevAttrMaxSharedMemoryPerBlock
    cuda_dev_attr_max_shared = 8

    lib_names = [
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "libcudart.dylib",
        "cudart64_120.dll",
        "cudart64_110.dll",
    ]

    cudart = None
    for name in lib_names:
        try:
            cudart = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if cudart is None:
        return None

    cuda_set_device = cudart.cudaSetDevice
    cuda_set_device.argtypes = [ctypes.c_int]
    cuda_set_device.restype = ctypes.c_int

    cuda_get_attr = cudart.cudaDeviceGetAttribute
    cuda_get_attr.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    cuda_get_attr.restype = ctypes.c_int

    # Ignore failure here and let attribute query decide availability.
    cuda_set_device(int(device))

    val_optin = ctypes.c_int()
    rc_optin = cuda_get_attr(ctypes.byref(val_optin), cuda_dev_attr_max_shared_optin, int(device))

    val_basic = ctypes.c_int()
    rc_basic = cuda_get_attr(ctypes.byref(val_basic), cuda_dev_attr_max_shared, int(device))

    candidates = []
    if rc_optin == 0 and val_optin.value > 0:
        candidates.append(val_optin.value)
    if rc_basic == 0 and val_basic.value > 0:
        candidates.append(val_basic.value)

    if not candidates:
        return None
    return max(candidates)


def maybe_compile(args: argparse.Namespace) -> None:
    binary_path = pathlib.Path(args.binary).resolve()
    source_path = pathlib.Path(args.source).resolve()

    mode = args.compile_mode
    if mode == "always":
        need_compile = True
    elif mode == "if_missing":
        need_compile = not binary_path.exists()
    elif mode == "never":
        need_compile = False
    else:
        raise ValueError(f"Unknown compile mode: {mode}")

    if not need_compile:
        return

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    cmd = [args.nvcc]
    if args.arch:
        cmd.append(f"-arch={args.arch}")
    cmd.extend(["-O3", "-std=c++20", str(source_path), "-o", str(binary_path)])
    if args.extra_nvcc_flags:
        cmd.extend(shlex.split(args.extra_nvcc_flags))

    print(f"[compile] {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Compile failed with exit code {proc.returncode}")


TMA_RE = re.compile(r"\[TMA\]\s+avg=([0-9.]+)\s+(ms|us)\s+BW=([0-9.]+)\s+GB/s\s+wrong=(\d+)")
SIMPLE_RE = re.compile(r"\[SIMPLE\]\s+avg=([0-9.]+)\s+(ms|us)\s+BW=([0-9.]+)\s+GB/s\s+wrong=(\d+)")
TMA_DETAIL_RE = re.compile(
    r"detail\(wait/issue/store\)\s+ns\s+=\s+([0-9.]+)\s+/\s+([0-9.]+)\s+/\s+([0-9.]+)\s+samples\s+=\s+(\d+)\s+/\s+(\d+)\s+/\s+(\d+)\s+blocks=(\d+)"
)
SIMPLE_DETAIL_RE = re.compile(
    r"detail\(io\)\s+ns\s+=\s+([0-9.]+)\s+samples\s+=\s+(\d+)\s+blocks=(\d+)"
)


def parse_output(text: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "tma_us": None,
        "tma_ms": None,
        "tma_bw_gbps": None,
        "tma_wrong": None,
        "simple_us": None,
        "simple_ms": None,
        "simple_bw_gbps": None,
        "simple_wrong": None,
        "tma_wait_ns": None,
        "tma_issue_ns": None,
        "tma_store_ns": None,
        "simple_io_ns": None,
    }

    m = TMA_RE.search(text)
    if m:
        t = float(m.group(1))
        unit = m.group(2)
        out["tma_us"] = t if unit == "us" else t * 1000.0
        out["tma_ms"] = t if unit == "ms" else t / 1000.0
        out["tma_bw_gbps"] = float(m.group(3))
        out["tma_wrong"] = float(m.group(4))

    m = SIMPLE_RE.search(text)
    if m:
        t = float(m.group(1))
        unit = m.group(2)
        out["simple_us"] = t if unit == "us" else t * 1000.0
        out["simple_ms"] = t if unit == "ms" else t / 1000.0
        out["simple_bw_gbps"] = float(m.group(3))
        out["simple_wrong"] = float(m.group(4))

    m = TMA_DETAIL_RE.search(text)
    if m:
        out["tma_wait_ns"] = float(m.group(1))
        out["tma_issue_ns"] = float(m.group(2))
        out["tma_store_ns"] = float(m.group(3))

    m = SIMPLE_DETAIL_RE.search(text)
    if m:
        out["simple_io_ns"] = float(m.group(1))

    return out


def fmt_cmd(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def safe_mean(vals: List[float]) -> Optional[float]:
    return statistics.mean(vals) if vals else None


def safe_stdev(vals: List[float]) -> Optional[float]:
    if len(vals) <= 1:
        return 0.0 if vals else None
    return statistics.stdev(vals)


def write_csv(path: pathlib.Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def markdown_table(rows: List[Dict[str, object]], columns: List[str]) -> str:
    if not rows:
        return "_(none)_"
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        vals = []
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run search space for simple_tma_transfer_bench and summarize results."
    )

    parser.add_argument("--binary", default="./simple_tma_transfer_bench")
    parser.add_argument("--source", default="./simple_tma_transfer_bench.cu")
    parser.add_argument("--compile-mode", choices=["never", "if_missing", "always"], default="if_missing")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", default="sm_90a")
    parser.add_argument("--extra-nvcc-flags", default="")

    parser.add_argument("--total-mb", default="64,128,256,512", help="list/range, e.g. 64,128 or 64:512:64")
    parser.add_argument("--grid", default="32")
    parser.add_argument("--block", default="544")
    parser.add_argument(
        "--smem-kb",
        default="64,96,100,128,150,192,200,300",
        help="Used only with --no-derive-smem. Otherwise ignored.",
    )
    parser.add_argument("--pipe", default="2,3")
    parser.add_argument("--tma-issue-warp", type=int, default=1, help="0 or 1; pass to --tma_issue_warp")
    parser.add_argument(
        "--derive-smem",
        dest="derive_smem",
        action="store_true",
        default=True,
        help="Set smem_kb automatically as tma_tile_kb * pipe (default on).",
    )
    parser.add_argument(
        "--no-derive-smem",
        dest="derive_smem",
        action="store_false",
        help="Use explicit --smem-kb list as independent search dimension.",
    )

    parser.add_argument("--simple-slice-kb", default="1024,2048")
    parser.add_argument("--tma-slice-kb", default="1024,2048")
    parser.add_argument("--tma-tile-kb", default="32,50,64,100")
    parser.add_argument("--pair-slices", dest="pair_slices", action="store_true", default=True,
                        help="Only keep configs where simple_slice_kb == tma_slice_kb (default on)")
    parser.add_argument("--no-pair-slices", dest="pair_slices", action="store_false")

    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--src", type=int, default=0)
    parser.add_argument("--dst", type=int, default=1)
    parser.add_argument("--detail", type=int, default=1)
    parser.add_argument("--verify", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--max-dyn-smem-kb",
        type=int,
        default=0,
        help="Override src GPU max dynamic shared memory (KB). 0 means auto-detect.",
    )

    parser.add_argument("--out-dir", default="")
    parser.add_argument("--env", action="append", default=[], help="Extra env in KEY=VALUE form")

    args = parser.parse_args()

    if args.repeats <= 0:
        print("--repeats must be > 0", file=sys.stderr)
        return 2
    if args.tma_issue_warp not in (0, 1):
        print("--tma-issue-warp must be 0 or 1", file=sys.stderr)
        return 2

    args.total_mb_list = parse_int_list(args.total_mb, "total-mb")
    args.grid_list = parse_int_list(args.grid, "grid")
    args.block_list = parse_int_list(args.block, "block")
    if args.derive_smem:
        args.smem_kb_list = []
    else:
        args.smem_kb_list = parse_int_list(args.smem_kb, "smem-kb")
    args.pipe_list = parse_int_list(args.pipe, "pipe")
    args.simple_slice_kb_list = parse_int_list(args.simple_slice_kb, "simple-slice-kb")
    args.tma_slice_kb_list = parse_int_list(args.tma_slice_kb, "tma-slice-kb")
    args.tma_tile_kb_list = parse_int_list(args.tma_tile_kb, "tma-tile-kb")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(f"transfer_search_{ts}{tag}")
    out_dir = out_dir.resolve()
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    extra_env: Dict[str, str] = {}
    for kv in args.env:
        if "=" not in kv:
            print(f"Invalid --env value (must be KEY=VALUE): {kv}", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        extra_env[k] = v

    configs, skipped = generate_configs(args)

    if args.max_dyn_smem_kb > 0:
        max_dyn_smem_bytes = args.max_dyn_smem_kb * 1024
    else:
        max_dyn_smem_bytes = query_max_dynamic_smem_bytes(args.src)
    if max_dyn_smem_bytes is not None:
        kept: List[BenchConfig] = []
        for cfg in configs:
            if cfg.smem_bytes > max_dyn_smem_bytes:
                skipped.append(
                    {
                        "total_mb": str(cfg.total_mb),
                        "grid": str(cfg.grid),
                        "block": str(cfg.block),
                        "smem_kb": str(cfg.smem_kb),
                        "pipe": str(cfg.pipe),
                        "simple_slice_kb": str(cfg.simple_slice_kb),
                        "tma_slice_kb": str(cfg.tma_slice_kb),
                        "tma_tile_kb": str(cfg.tma_tile_kb),
                        "reason": f"smem_bytes > device_max_dynamic_smem ({cfg.smem_bytes} > {max_dyn_smem_bytes})",
                    }
                )
            else:
                kept.append(cfg)
        configs = kept

    planned_path = out_dir / "planned_configs.csv"
    write_csv(
        planned_path,
        [dataclasses.asdict(c) for c in configs],
        [
            "total_mb", "grid", "block", "smem_kb", "pipe",
            "simple_slice_kb", "tma_slice_kb", "tma_tile_kb",
        ],
    )

    if skipped:
        skipped_path = out_dir / "skipped_configs.csv"
        write_csv(
            skipped_path,
            skipped,
            [
                "total_mb", "grid", "block", "smem_kb", "pipe",
                "simple_slice_kb", "tma_slice_kb", "tma_tile_kb", "reason",
            ],
        )

    print(f"[plan] valid configs={len(configs)}, skipped={len(skipped)}, repeats={args.repeats}")
    print(f"[plan] pair_slices={args.pair_slices}")
    print("[plan] TMA constraint: slot_bytes == tile_bytes (smem_bytes/pipe == tma_tile_bytes)")
    print(f"[plan] derive_smem={args.derive_smem}")
    if max_dyn_smem_bytes is not None:
        if args.max_dyn_smem_kb > 0:
            print(
                f"[plan] src device max dynamic smem: {max_dyn_smem_bytes} B "
                f"({max_dyn_smem_bytes // 1024} KB, user override)"
            )
        else:
            print(f"[plan] src device max dynamic smem: {max_dyn_smem_bytes} B ({max_dyn_smem_bytes // 1024} KB)")
    else:
        print("[plan] src device max dynamic smem: unknown (could not query with cudart)")
    print(f"[plan] output dir: {out_dir}")

    if args.dry_run:
        print("[dry-run] no execution")
        return 0

    try:
        maybe_compile(args)
    except Exception as e:
        print(f"[error] compile step failed: {e}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.update(extra_env)

    all_rows: List[Dict[str, object]] = []
    total_runs = len(configs) * args.repeats
    run_idx = 0
    start_all = time.time()
    bin_path = str(pathlib.Path(args.binary).resolve())

    for cfg_idx, cfg in enumerate(configs):
        for rep in range(args.repeats):
            run_idx += 1
            cmd = [
                bin_path,
                "--total_mb", str(cfg.total_mb),
                "--grid", str(cfg.grid),
                "--block", str(cfg.block),
                "--iters", str(args.iters),
                "--warmup", str(args.warmup),
                "--src", str(args.src),
                "--dst", str(args.dst),
                "--proto", "0",
                "--smem_kb", str(cfg.smem_kb),
                "--pipe", str(cfg.pipe),
                "--tma_issue_warp", str(args.tma_issue_warp),
                "--simple_slice_kb", str(cfg.simple_slice_kb),
                "--tma_slice_kb", str(cfg.tma_slice_kb),
                "--tma_tile_kb", str(cfg.tma_tile_kb),
                "--detail", str(args.detail),
                "--verify", str(args.verify),
            ]
            cmd_str = fmt_cmd(cmd)
            print(
                f"[run {run_idx}/{total_runs}] cfg={cfg_idx+1}/{len(configs)} "
                f"(mb={cfg.total_mb},g={cfg.grid},b={cfg.block},sm={cfg.smem_kb},p={cfg.pipe},"
                f"ss={cfg.simple_slice_kb},ts={cfg.tma_slice_kb},tt={cfg.tma_tile_kb},rep={rep+1})"
            )

            t0 = time.time()
            status = "ok"
            returncode = 0
            output = ""
            error_message = ""

            try:
                proc = subprocess.run(
                    cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=args.timeout_sec,
                )
                output = proc.stdout
                returncode = proc.returncode
                if returncode != 0:
                    status = "failed"
            except subprocess.TimeoutExpired as e:
                status = "timeout"
                returncode = 124
                output = (e.stdout or "") + "\n[timeout]"
                error_message = f"timeout after {args.timeout_sec}s"
            except Exception as e:
                status = "failed"
                returncode = 1
                error_message = str(e)
                output = f"[exception] {e}"

            elapsed_s = time.time() - t0
            parsed = parse_output(output)

            if status == "ok" and (parsed["tma_us"] is None or parsed["simple_us"] is None):
                status = "parse_error"
                error_message = "missing [TMA]/[SIMPLE] lines in output"

            speedup_bw = None
            speedup_us = None
            if parsed["tma_bw_gbps"] and parsed["simple_bw_gbps"] and parsed["simple_bw_gbps"] > 0:
                speedup_bw = parsed["tma_bw_gbps"] / parsed["simple_bw_gbps"]
            if parsed["tma_us"] and parsed["simple_us"] and parsed["tma_us"] > 0:
                speedup_us = parsed["simple_us"] / parsed["tma_us"]

            log_name = (
                f"run_{run_idx:05d}_mb{cfg.total_mb}_g{cfg.grid}_b{cfg.block}_"
                f"sm{cfg.smem_kb}_p{cfg.pipe}_ss{cfg.simple_slice_kb}_"
                f"ts{cfg.tma_slice_kb}_tt{cfg.tma_tile_kb}_r{rep+1}.log"
            )
            log_path = logs_dir / log_name
            with log_path.open("w") as f:
                f.write(f"# cmd: {cmd_str}\n")
                f.write(f"# status: {status}\n")
                if error_message:
                    f.write(f"# error: {error_message}\n")
                f.write(output)

            row: Dict[str, object] = {
                "run_index": run_idx,
                "status": status,
                "returncode": returncode,
                "error": error_message,
                "elapsed_s": elapsed_s,
                "cmd": cmd_str,
                "log_path": str(log_path),
                "repeat": rep + 1,

                "total_mb": cfg.total_mb,
                "grid": cfg.grid,
                "block": cfg.block,
                "smem_kb": cfg.smem_kb,
                "pipe": cfg.pipe,
                "tma_issue_warp": args.tma_issue_warp,
                "simple_slice_kb": cfg.simple_slice_kb,
                "tma_slice_kb": cfg.tma_slice_kb,
                "tma_tile_kb": cfg.tma_tile_kb,

                "tma_us": parsed["tma_us"],
                "tma_ms": parsed["tma_ms"],
                "tma_bw_gbps": parsed["tma_bw_gbps"],
                "tma_wrong": parsed["tma_wrong"],
                "simple_us": parsed["simple_us"],
                "simple_ms": parsed["simple_ms"],
                "simple_bw_gbps": parsed["simple_bw_gbps"],
                "simple_wrong": parsed["simple_wrong"],
                "speedup_bw_tma_over_simple": speedup_bw,
                "speedup_us_simple_over_tma": speedup_us,
                "tma_wait_ns": parsed["tma_wait_ns"],
                "tma_issue_ns": parsed["tma_issue_ns"],
                "tma_store_ns": parsed["tma_store_ns"],
                "simple_io_ns": parsed["simple_io_ns"],
            }
            all_rows.append(row)

    raw_path = out_dir / "raw_runs.csv"
    write_csv(
        raw_path,
        all_rows,
        [
            "run_index", "status", "returncode", "error", "elapsed_s",
            "repeat", "total_mb", "grid", "block", "smem_kb", "pipe", "simple_slice_kb", "tma_slice_kb", "tma_tile_kb",
            "tma_issue_warp",
            "tma_us", "tma_ms", "tma_bw_gbps", "tma_wrong",
            "simple_us", "simple_ms", "simple_bw_gbps", "simple_wrong",
            "speedup_bw_tma_over_simple", "speedup_us_simple_over_tma",
            "tma_wait_ns", "tma_issue_ns", "tma_store_ns", "simple_io_ns",
            "cmd", "log_path",
        ],
    )

    grouped: Dict[Tuple[int, int, int, int, int, int, int, int], List[Dict[str, object]]] = {}
    for row in all_rows:
        key = (
            int(row["total_mb"]),
            int(row["grid"]),
            int(row["block"]),
            int(row["smem_kb"]),
            int(row["pipe"]),
            int(row["simple_slice_kb"]),
            int(row["tma_slice_kb"]),
            int(row["tma_tile_kb"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for key, rows in grouped.items():
        ok_rows = [r for r in rows if r["status"] == "ok"]
        if not ok_rows:
            continue

        tma_us = [float(r["tma_us"]) for r in ok_rows if r["tma_us"] is not None]
        tma_bw = [float(r["tma_bw_gbps"]) for r in ok_rows if r["tma_bw_gbps"] is not None]
        simple_us = [float(r["simple_us"]) for r in ok_rows if r["simple_us"] is not None]
        simple_bw = [float(r["simple_bw_gbps"]) for r in ok_rows if r["simple_bw_gbps"] is not None]
        tma_wrong = [float(r["tma_wrong"]) for r in ok_rows if r["tma_wrong"] is not None]
        simple_wrong = [float(r["simple_wrong"]) for r in ok_rows if r["simple_wrong"] is not None]

        tma_us_mean = safe_mean(tma_us)
        tma_bw_mean = safe_mean(tma_bw)
        simple_us_mean = safe_mean(simple_us)
        simple_bw_mean = safe_mean(simple_bw)

        speedup_bw = None
        speedup_us = None
        if tma_bw_mean is not None and simple_bw_mean not in (None, 0.0):
            speedup_bw = tma_bw_mean / simple_bw_mean
        if tma_us_mean not in (None, 0.0) and simple_us_mean is not None:
            speedup_us = simple_us_mean / tma_us_mean

        summary_rows.append(
            {
                "total_mb": key[0],
                "grid": key[1],
                "block": key[2],
                "smem_kb": key[3],
                "pipe": key[4],
                "tma_issue_warp": args.tma_issue_warp,
                "simple_slice_kb": key[5],
                "tma_slice_kb": key[6],
                "tma_tile_kb": key[7],

                "runs_total": len(rows),
                "runs_ok": len(ok_rows),
                "runs_not_ok": len(rows) - len(ok_rows),
                "tma_wrong_max": max(tma_wrong) if tma_wrong else None,
                "simple_wrong_max": max(simple_wrong) if simple_wrong else None,

                "tma_us_mean": tma_us_mean,
                "tma_us_std": safe_stdev(tma_us),
                "tma_bw_gbps_mean": tma_bw_mean,
                "tma_bw_gbps_std": safe_stdev(tma_bw),
                "simple_us_mean": simple_us_mean,
                "simple_us_std": safe_stdev(simple_us),
                "simple_bw_gbps_mean": simple_bw_mean,
                "simple_bw_gbps_std": safe_stdev(simple_bw),
                "speedup_bw_tma_over_simple": speedup_bw,
                "speedup_us_simple_over_tma": speedup_us,
            }
        )

    summary_rows.sort(
        key=lambda r: (
            -1.0 if r["speedup_bw_tma_over_simple"] is None else -float(r["speedup_bw_tma_over_simple"]),
            -1.0 if r["tma_bw_gbps_mean"] is None else -float(r["tma_bw_gbps_mean"]),
        )
    )

    summary_path = out_dir / "summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "total_mb", "grid", "block", "smem_kb", "pipe", "simple_slice_kb", "tma_slice_kb", "tma_tile_kb",
            "tma_issue_warp",
            "runs_total", "runs_ok", "runs_not_ok", "tma_wrong_max", "simple_wrong_max",
            "tma_us_mean", "tma_us_std", "tma_bw_gbps_mean", "tma_bw_gbps_std",
            "simple_us_mean", "simple_us_std", "simple_bw_gbps_mean", "simple_bw_gbps_std",
            "speedup_bw_tma_over_simple", "speedup_us_simple_over_tma",
        ],
    )

    done_s = time.time() - start_all
    ok_total = sum(1 for r in all_rows if r["status"] == "ok")
    fail_total = len(all_rows) - ok_total

    clean_summary = [
        r for r in summary_rows
        if (r["tma_wrong_max"] in (None, 0.0))
        and (r["simple_wrong_max"] in (None, 0.0))
    ]

    top_bw = sorted(
        clean_summary,
        key=lambda r: -float(r["tma_bw_gbps_mean"]) if r["tma_bw_gbps_mean"] is not None else math.inf,
    )[:10]

    top_speedup = sorted(
        clean_summary,
        key=lambda r: -float(r["speedup_bw_tma_over_simple"]) if r["speedup_bw_tma_over_simple"] is not None else math.inf,
    )[:10]

    md_path = out_dir / "summary.md"
    with md_path.open("w") as f:
        f.write("# Transfer Search Summary\n\n")
        f.write(f"- Timestamp: {dt.datetime.now().isoformat()}\n")
        f.write(f"- Output dir: `{out_dir}`\n")
        f.write(f"- Total configs: {len(configs)}\n")
        f.write(f"- Repeats: {args.repeats}\n")
        f.write(f"- Pair slices: {args.pair_slices}\n")
        f.write(f"- Total runs: {len(all_rows)}\n")
        f.write(f"- Runs ok: {ok_total}\n")
        f.write(f"- Runs not ok: {fail_total}\n")
        f.write("- Time unit in summary tables: us\n")
        f.write(f"- Total wall time: {done_s:.1f} s\n")
        f.write("\n")

        f.write("## Search Space\n\n")
        f.write(f"- total_mb: {args.total_mb_list}\n")
        f.write(f"- grid: {args.grid_list}\n")
        f.write(f"- block: {args.block_list}\n")
        if args.derive_smem:
            f.write("- smem_kb: derived as tma_tile_kb * pipe\n")
        else:
            f.write(f"- smem_kb: {args.smem_kb_list}\n")
        f.write(f"- pipe: {args.pipe_list}\n")
        f.write(f"- tma_issue_warp: {args.tma_issue_warp}\n")
        f.write(f"- simple_slice_kb: {args.simple_slice_kb_list}\n")
        f.write(f"- tma_slice_kb: {args.tma_slice_kb_list}\n")
        f.write(f"- tma_tile_kb: {args.tma_tile_kb_list}\n")
        f.write("\n")

        f.write("## Top 10 by TMA BW\n\n")
        f.write(
            markdown_table(
                top_bw,
                [
                    "total_mb", "grid", "block", "smem_kb", "pipe",
                    "tma_issue_warp",
                    "simple_slice_kb", "tma_slice_kb", "tma_tile_kb",
                    "tma_bw_gbps_mean", "simple_bw_gbps_mean", "speedup_bw_tma_over_simple",
                    "tma_us_mean", "simple_us_mean",
                ],
            )
        )
        f.write("\n\n")

        f.write("## Top 10 by Speedup (TMA BW / Simple BW)\n\n")
        f.write(
            markdown_table(
                top_speedup,
                [
                    "total_mb", "grid", "block", "smem_kb", "pipe",
                    "tma_issue_warp",
                    "simple_slice_kb", "tma_slice_kb", "tma_tile_kb",
                    "speedup_bw_tma_over_simple", "tma_bw_gbps_mean", "simple_bw_gbps_mean",
                    "tma_us_mean", "simple_us_mean",
                ],
            )
        )
        f.write("\n")

    print("[done]")
    print(f"  raw runs   : {raw_path}")
    print(f"  summary csv: {summary_path}")
    print(f"  summary md : {md_path}")
    if skipped:
        print(f"  skipped cfg: {out_dir / 'skipped_configs.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
