#!/usr/bin/env python3
"""Run and plot BlockToeplitzInverse nsys component benchmarks.

The benchmarked C++ test reads:

  BTI_BENCH_T
  BTI_BENCH_R
  BTI_BENCH_COEFF_SCALE

and profiles the preloaded Newton solve via cudaProfilerStart/Stop.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
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
    "Pack": "#4C78A8",
    "FFT": "#F2CF5B",
    "Transpose": "#D37295",
    "GEMM": "#8E63B0",
    "Build V": "#59A14F",
    "Unpack": "#C49A3A",
    "Other": "#B8B8B8",
}

NEWTON_RANGE_RE = re.compile(r":?newton_(\d+)_to_(\d+)_fft_(\d+)$")


def parse_int_values(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
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


def write_csv(
    csv_path: Path,
    summaries: dict[tuple[int, int], dict[str, float]],
) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T", "block_dim", "component", "time_ms", "percent"])
        for t_value, block_dim in sorted(summaries):
            total = sum(summaries[(t_value, block_dim)].values())
            for component in COMPONENT_ORDER:
                time_ms = summaries[(t_value, block_dim)].get(component, 0.0)
                percent = 100.0 * time_ms / total if total > 0.0 else 0.0
                writer.writerow([t_value, block_dim, component, f"{time_ms:.6f}", f"{percent:.6f}"])


def summarize_iterations(
    *,
    nsys: str,
    sqlite_path: Path,
) -> list[tuple[int, int, int, float]]:
    """Return per-Newton-iteration GPU kernel time from nsys nvtx_kern_sum.

    Nsight Systems associates kernels with the CPU-side NVTX range that launched
    them. Reusing nsys' report avoids guessing that association from raw sqlite
    timestamps, which is brittle because launches and GPU execution are async.
    """
    cmd = [nsys, "stats", "--report", "nvtx_kern_sum", str(sqlite_path)]
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    totals_ms: dict[tuple[int, int, int], float] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith(":newton_") and not line.startswith("newton_"):
            continue

        fields = line.split()
        if len(fields) < 7:
            continue

        match = NEWTON_RANGE_RE.match(fields[0])
        if match is None:
            continue

        try:
            duration_ns = int(fields[6].replace(",", ""))
        except ValueError:
            continue

        key = tuple(int(value) for value in match.groups())
        totals_ms[key] = totals_ms.get(key, 0.0) + duration_ns / 1.0e6

    if not totals_ms:
        raise RuntimeError(
            "Could not find per-iteration NVTX kernel rows. "
            "Make sure the report was collected with --trace=cuda,cublas,nvtx."
        )

    return [
        (m, m_next, fft_len, time_ms)
        for (m, m_next, fft_len), time_ms in sorted(totals_ms.items())
    ]


def write_iteration_csv(
    csv_path: Path,
    *,
    t_value: int,
    block_dim: int,
    iterations: list[tuple[int, int, int, float]],
) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T", "block_dim", "iteration", "m", "m_next", "fft_len", "time_ms"])
        for index, (m, m_next, fft_len, time_ms) in enumerate(iterations, start=1):
            writer.writerow(
                [
                    t_value,
                    block_dim,
                    index,
                    m,
                    m_next,
                    fft_len,
                    f"{time_ms:.6f}",
                ]
            )


def plot_iterations(
    png_path: Path,
    *,
    t_value: int,
    block_dim: int,
    iterations: list[tuple[int, int, int, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
        }
    )

    labels = [f"{m}->{m_next}" for m, m_next, _, _ in iterations]
    values = [time_ms for _, _, _, time_ms in iterations]
    x = list(range(len(iterations)))

    fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
    bars = ax.bar(
        x,
        values,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.55,
        width=0.52,
    )

    y_max = max(values) if values else 0.0
    ax.set_ylim(0.0, y_max * 1.18 if y_max > 0.0 else 1.0)
    for bar, value in zip(bars, values):
        label = f"{value:.1f}" if value < 100.0 else f"{value:.0f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.025 * y_max,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Block Toeplitz inverse iterations (T = {t_value}, r = {block_dim})")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(png_path, dpi=220)
    fig.savefig(png_path.with_suffix(".pdf"))
    fig.savefig(png_path.with_suffix(".svg"))


def plot_components(
    png_path: Path,
    summaries: dict[int, dict[str, float]],
    *,
    xlabel: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
        }
    )

    x_values = sorted(summaries)
    x = list(range(len(x_values)))
    bottoms = [0.0] * len(x_values)
    totals = [sum(summaries[value].values()) for value in x_values]

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    for component in COMPONENT_ORDER:
        values = [summaries[value].get(component, 0.0) for value in x_values]
        if component == "Other" and max(values, default=0.0) < 1e-6:
            continue
        ax.bar(
            x,
            values,
            bottom=bottoms,
            label=component,
            color=COMPONENT_COLORS[component],
            edgecolor="white",
            linewidth=0.45,
            width=0.48,
        )
        bottoms = [base + value for base, value in zip(bottoms, values)]

    y_max = max(totals) if totals else 0.0
    ax.set_ylim(0.0, y_max * 1.14 if y_max > 0.0 else 1.0)
    for x_pos, total in zip(x, totals):
        label = f"{total:.0f}" if total >= 100.0 else f"{total:.1f}"
        ax.text(
            x_pos,
            total + 0.025 * y_max,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in x_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(
        title="Component",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("#FAFAFA")
    fig.savefig(png_path, dpi=220)
    fig.savefig(png_path.with_suffix(".pdf"))
    fig.savefig(png_path.with_suffix(".svg"))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-values", default="1024,2048,3072,4096,5120")
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--r-values", default="")
    parser.add_argument("--fixed-t", type=int, default=4096)
    parser.add_argument(
        "--iteration-plot",
        action="store_true",
        help="Plot total GPU kernel time for each Newton iteration of one case.",
    )
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

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.iteration_plot:
        t_value = args.fixed_t
        block_dim = args.block_dim
        output_stem = output_dir / f"bti_T{t_value}_r{block_dim}"
        if not args.no_run:
            run_profile(
                nsys=args.nsys,
                test_exe=args.test_exe.resolve(),
                output_stem=output_stem,
                t_value=t_value,
                block_dim=block_dim,
                coeff_scale=args.coeff_scale,
                repo_root=repo_root,
            )

        sqlite_path = output_dir / f"bti_T{t_value}_r{block_dim}.sqlite"
        if not sqlite_path.exists():
            raise FileNotFoundError(f"Missing nsys sqlite file: {sqlite_path}")

        iterations = summarize_iterations(nsys=args.nsys, sqlite_path=sqlite_path)
        csv_path = output_dir / f"bti_iterations_T{t_value}_r{block_dim}.csv"
        png_path = output_dir / f"bti_iterations_T{t_value}_r{block_dim}.png"
        write_iteration_csv(
            csv_path,
            t_value=t_value,
            block_dim=block_dim,
            iterations=iterations,
        )
        plot_iterations(
            png_path,
            t_value=t_value,
            block_dim=block_dim,
            iterations=iterations,
        )

        print(f"\nWrote {csv_path}")
        print(f"Wrote {png_path}")
        print(f"Wrote {png_path.with_suffix('.pdf')}")
        print(f"Wrote {png_path.with_suffix('.svg')}")
        return

    r_values = parse_int_values(args.r_values) if args.r_values else []
    if r_values:
        cases = [(args.fixed_t, r_value) for r_value in r_values]
        mode = "r"
    else:
        cases = [(t_value, args.block_dim) for t_value in parse_int_values(args.t_values)]
        mode = "t"

    for t_value, block_dim in cases:
        output_stem = output_dir / f"bti_T{t_value}_r{block_dim}"
        if not args.no_run:
            run_profile(
                nsys=args.nsys,
                test_exe=args.test_exe.resolve(),
                output_stem=output_stem,
                t_value=t_value,
                block_dim=block_dim,
                coeff_scale=args.coeff_scale,
                repo_root=repo_root,
            )

    case_summaries: dict[tuple[int, int], dict[str, float]] = {}
    for t_value, block_dim in cases:
        sqlite_path = output_dir / f"bti_T{t_value}_r{block_dim}.sqlite"
        if not sqlite_path.exists():
            raise FileNotFoundError(f"Missing nsys sqlite file: {sqlite_path}")
        case_summaries[(t_value, block_dim)] = summarize_sqlite(sqlite_path)

    if mode == "r":
        plot_summaries = {block_dim: case_summaries[(args.fixed_t, block_dim)] for block_dim in r_values}
        csv_path = output_dir / f"bti_components_T{args.fixed_t}.csv"
        png_path = output_dir / f"bti_components_T{args.fixed_t}.png"
        xlabel = "Block dimension r"
        title = f"Block Toeplitz inverse components (T = {args.fixed_t})"
    else:
        t_values = [t_value for t_value, _ in cases]
        plot_summaries = {t_value: case_summaries[(t_value, args.block_dim)] for t_value in t_values}
        csv_path = output_dir / f"bti_components_r{args.block_dim}.csv"
        png_path = output_dir / f"bti_components_r{args.block_dim}.png"
        xlabel = "Number of block coefficients T"
        title = f"Block Toeplitz inverse components (r = {args.block_dim})"

    write_csv(csv_path, case_summaries)
    plot_components(png_path, plot_summaries, xlabel=xlabel, title=title)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {png_path.with_suffix('.pdf')}")
    print(f"Wrote {png_path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
