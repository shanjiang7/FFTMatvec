#!/usr/bin/env python3
"""Run and plot BlockToeplitzInverse nsys component benchmarks.

The benchmarked C++ test reads:

  BTI_BENCH_T
  BTI_BENCH_R
  BTI_BENCH_COEFF_SCALE

and profiles only the preloaded Newton hot path via cudaProfilerStart/Stop.
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import subprocess
from pathlib import Path


COMPONENT_ORDER = [
    "Pack",
    "FFT",
    "Transpose",
    "GEMM",
    "Build V",
    "Unpack",
    "Other",
]

COMPONENT_COLORS = {
    "Pack": "#4E79A7",
    "FFT": "#F1CE63",
    "Transpose": "#D37295",
    "GEMM": "#9D5DBA",
    "Build V": "#59A14F",
    "Unpack": "#C49A3A",
    "Other": "#BAB0AC",
}


def parse_t_values(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one T value is required.")
    return values


def component_for_kernel(name: str) -> str:
    lower = name.lower()
    if "pack_blocks_to_entry_real" in lower:
        return "Pack"
    if "unpack_entry_real_to_blocks" in lower:
        return "Unpack"
    if "build_newton_v_real" in lower:
        return "Build V"
    if "gemm" in lower or "xmma" in lower:
        return "GEMM"
    if "transpose" in lower:
        return "Transpose"
    if "fft" in lower or "preprocess_kernel" in lower or "postprocess_kernel" in lower:
        return "FFT"
    return "Other"


def sqlite_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row[0] for row in rows}


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def kernel_rows(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    tables = sqlite_tables(conn)
    table = None
    for candidate in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUDA_GPU_KERNEL", "GPU_KERNEL"):
        if candidate in tables:
            table = candidate
            break
    if table is None:
        raise RuntimeError("Could not find an nsys GPU kernel table in sqlite file.")

    cols = table_columns(conn, table)
    if "start" in cols and "end" in cols:
        duration_expr = '(k."end" - k."start")'
    elif "duration" in cols:
        duration_expr = 'k."duration"'
    elif "totalTime" in cols:
        duration_expr = 'k."totalTime"'
    else:
        raise RuntimeError(f"Could not find a duration column in {table}.")

    name_candidates = [
        col
        for col in ("demangledName", "shortName", "mangledName", "name")
        if col in cols
    ]
    if not name_candidates:
        raise RuntimeError(f"Could not find a kernel name column in {table}.")

    joins: list[str] = []
    name_exprs: list[str] = []
    if "StringIds" in tables:
        string_cols = table_columns(conn, "StringIds")
        if {"id", "value"}.issubset(string_cols):
            for idx, col in enumerate(name_candidates):
                alias = f"s{idx}"
                joins.append(f'LEFT JOIN StringIds {alias} ON k."{col}" = {alias}.id')
                name_exprs.append(f"{alias}.value")
                name_exprs.append(f'CAST(k."{col}" AS TEXT)')

    if not name_exprs:
        name_exprs = [f'CAST(k."{col}" AS TEXT)' for col in name_candidates]

    name_expr = "COALESCE(" + ", ".join(name_exprs) + ")"
    query = f"""
        SELECT {duration_expr} AS duration_ns, {name_expr} AS kernel_name
        FROM {table} k
        {' '.join(joins)}
        WHERE {duration_expr} IS NOT NULL
    """
    rows = conn.execute(query).fetchall()
    return [(int(row[0]), str(row[1])) for row in rows if row[0] is not None]


def summarize_sqlite(sqlite_path: Path) -> dict[str, float]:
    totals = {component: 0.0 for component in COMPONENT_ORDER}
    with sqlite3.connect(sqlite_path) as conn:
        for duration_ns, kernel_name in kernel_rows(conn):
            totals[component_for_kernel(kernel_name)] += duration_ns / 1.0e6
    return totals


def run_profile(
    *,
    nsys: str,
    test_exe: Path,
    output_stem: Path,
    t_value: int,
    block_dim: int,
    coeff_scale: float,
    repo_root: Path,
) -> None:
    env = os.environ.copy()
    env["BTI_BENCH_T"] = str(t_value)
    env["BTI_BENCH_R"] = str(block_dim)
    env["BTI_BENCH_COEFF_SCALE"] = str(coeff_scale)

    cmd = [
        nsys,
        "profile",
        "--trace=cuda,cublas,nvtx",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--stats=true",
        "--force-overwrite=true",
        "-o",
        str(output_stem),
        str(test_exe),
        "--gtest_filter=BlockToeplitzInverseTest.BenchmarkNsysWarmLarge",
    ]
    print(f"\n[run] T={t_value}, r={block_dim}")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def write_csv(csv_path: Path, summaries: dict[int, dict[str, float]], block_dim: int) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T", "block_dim", "component", "time_ms", "percent"])
        for t_value in sorted(summaries):
            total = sum(summaries[t_value].values())
            for component in COMPONENT_ORDER:
                time_ms = summaries[t_value].get(component, 0.0)
                percent = 100.0 * time_ms / total if total > 0.0 else 0.0
                writer.writerow([t_value, block_dim, component, f"{time_ms:.6f}", f"{percent:.6f}"])


def plot_components(
    png_path: Path, summaries: dict[int, dict[str, float]], block_dim: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_values = sorted(summaries)
    x = list(range(len(t_values)))
    bottoms = [0.0] * len(t_values)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for component in COMPONENT_ORDER:
        values = [summaries[t].get(component, 0.0) for t in t_values]
        if component == "Other" and max(values, default=0.0) < 1e-6:
            continue
        ax.bar(
            x,
            values,
            bottom=bottoms,
            label=component,
            color=COMPONENT_COLORS[component],
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms = [base + value for base, value in zip(bottoms, values)]

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in t_values])
    ax.set_xlabel("T")
    ax.set_ylabel("Hot path time (ms)")
    ax.set_title(f"Block Toeplitz Inverse Hot Path Components (r={block_dim})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(png_path.with_suffix(".pdf"))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-values", default="1024,2048,3072,4096,5120")
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--coeff-scale", type=float, default=0.001)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument(
        "--test-exe",
        type=Path,
        default=repo_root / "build" / "Tests" / "BlockToeplitzInverseTest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "benchmark_results" / "block_toeplitz_inverse",
    )
    parser.add_argument("--no-run", action="store_true", help="Parse existing sqlite files only.")
    args = parser.parse_args()

    t_values = parse_t_values(args.t_values)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for t_value in t_values:
        output_stem = output_dir / f"bti_T{t_value}_r{args.block_dim}"
        if not args.no_run:
            run_profile(
                nsys=args.nsys,
                test_exe=args.test_exe.resolve(),
                output_stem=output_stem,
                t_value=t_value,
                block_dim=args.block_dim,
                coeff_scale=args.coeff_scale,
                repo_root=repo_root,
            )

    summaries: dict[int, dict[str, float]] = {}
    for t_value in t_values:
        sqlite_path = output_dir / f"bti_T{t_value}_r{args.block_dim}.sqlite"
        if not sqlite_path.exists():
            raise FileNotFoundError(f"Missing nsys sqlite file: {sqlite_path}")
        summaries[t_value] = summarize_sqlite(sqlite_path)

    csv_path = output_dir / f"bti_components_r{args.block_dim}.csv"
    png_path = output_dir / f"bti_components_r{args.block_dim}.png"
    write_csv(csv_path, summaries, args.block_dim)
    plot_components(png_path, summaries, args.block_dim)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {png_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
