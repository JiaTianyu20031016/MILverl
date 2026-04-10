import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
import numpy as np


STEP_RE = re.compile(r"step:(\d+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def parse_value(raw: str) -> Optional[float]:
    raw = raw.strip()
    raw = strip_ansi(raw)
    # unwrap np.float64(…) or similar wrappers
    if raw.startswith("np.float64(") and raw.endswith(")"):
        raw = raw[len("np.float64(") : -1]
    # remove potential trailing commas or artifacts
    raw = raw.strip().strip(",")
    try:
        return float(raw)
    except Exception:
        return None


def parse_step_line(line: str) -> Tuple[Optional[int], Dict[str, float]]:
    line = strip_ansi(line)
    if "step:" not in line:
        return None, {}
    # Keep only the part from 'step:' onward
    idx = line.find("step:")
    part = line[idx:]
    # Split pairs by ' - '
    chunks = [c for c in part.split(" - ") if c]
    if not chunks:
        return None, {}
    # First chunk should contain step
    m = STEP_RE.search(chunks[0])
    step = int(m.group(1)) if m else None
    metrics: Dict[str, float] = {}
    for ch in chunks:
        if ":" not in ch:
            continue
        k, v = ch.split(":", 1)
        k = k.strip()
        if k.startswith("step"):
            continue
        val = parse_value(v)
        if val is not None:
            metrics[k] = val
    return step, metrics


def load_metrics(log_path: str, keys: List[str]) -> Tuple[List[int], Dict[str, List[Optional[float]]]]:
    steps_seen: Dict[int, Dict[str, float]] = {}
    extra_keys: Set[str] = set()
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "step:" not in line:
                continue
            step, mets = parse_step_line(line)
            if step is None:
                continue
            filtered: Dict[str, float] = {}
            for k, v in mets.items():
                k_clean = strip_ansi(k.strip())
                if "val-core" in k_clean or "debug" in k_clean:
                    extra_keys.add(k_clean)
                    filtered[k_clean] = v
                elif k_clean in keys:
                    filtered[k_clean] = v
            # Merge (latest wins)
            if filtered:
                prev = steps_seen.get(step, {})
                prev.update(filtered)
                steps_seen[step] = prev

    if not steps_seen:
        raise SystemExit(f"No step lines found in {log_path}")

    all_keys = list(dict.fromkeys(keys + sorted(extra_keys)))
    steps_sorted = sorted(steps_seen.keys())
    out: Dict[str, List[Optional[float]]] = {k: [] for k in all_keys}
    for s in steps_sorted:
        met = steps_seen[s]
        for k in all_keys:
            out[k].append(met.get(k, None))
    return steps_sorted, out


def plot_lines(steps: List[int], series: Dict[str, List[Optional[float]]], output_dir: str, separate: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    keys = list(series.keys())
    # Convert to numpy with masking of None
    data = {}
    for k in keys:
        arr = np.array([np.nan if v is None else float(v) for v in series[k]], dtype=float)
        data[k] = arr

    def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
        if window is None or window <= 1:
            return arr.copy()
        x = arr.astype(float)
        mask = np.isfinite(x)
        k = np.ones(int(window), dtype=float)
        valid = np.convolve(mask.astype(float), k, mode="same")
        filled = np.where(mask, x, 0.0)
        summed = np.convolve(filled, k, mode="same")
        with np.errstate(invalid='ignore', divide='ignore'):
            out = summed / np.maximum(valid, 1e-12)
        out[valid == 0] = np.nan
        return out

    ma_window = getattr(plot_lines, "_ma_window", 1)

    # Subplots grid
    n = len(keys)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.0 * rows), squeeze=False)
    for i, k in enumerate(keys):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        y = data[k]
        # Plot raw (light), then smoothed overlay (bold)
        raw_line = ax.plot(steps, y, label=f"{k} (raw)", linewidth=1.0, alpha=0.35)[0]
        color = raw_line.get_color()
        y_ma = moving_average(y, ma_window)
        ax.plot(steps, y_ma, label=f"{k} (avg)", linewidth=2.0, color=color)
        ax.set_title(k)
        ax.set_xlabel("step")
        ax.set_ylabel(k)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    # Hide empty axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    out_path = os.path.join(output_dir, "metrics_over_steps.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")

    if separate:
        for k in keys:
            plt.figure(figsize=(8, 4))
            raw_line = plt.plot(steps, data[k], label=f"{k} (raw)", linewidth=1.0, alpha=0.35)[0]
            color = raw_line.get_color()
            y_ma = moving_average(data[k], ma_window)
            plt.plot(steps, y_ma, label=f"{k} (avg)", linewidth=2.0, color=color)
            plt.title(k)
            plt.xlabel("step")
            plt.ylabel(k)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=9)
            sp = os.path.join(output_dir, f"{k.replace('/', '_').replace('@', '_')}.png")
            plt.savefig(sp, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"[OK] Saved {sp}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics over steps from VERL logs")
    parser.add_argument("--log", type=str, required=True, help="Path to outlog.txt or similar training log")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save plots")
    parser.add_argument("--separate", action="store_true", help="Also save per-metric individual figures")
    parser.add_argument("--ma-window", type=int, default=11, help="Moving-average window size for trend lines; 1 disables smoothing")
    args = parser.parse_args()

    default_keys = [
        "actor/entropy",
        "actor/kl_loss",
        "actor/grad_norm",
        "critic/rewards/mean",
        "critic/advantages/mean",
        "critic/advantages/max",
        "critic/advantages/min",
        "response_length/mean",
        "response_length/clip_ratio",
        'critic/score/consistency_to_rule',
        'critic/score/var'
    ]

    steps, series = load_metrics(args.log, default_keys)
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.log), "plots")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Pass window via function attribute to avoid changing signature across code
    plot_lines._ma_window = max(1, int(args.ma_window))
    plot_lines(steps, series, out_dir, separate=args.separate)


if __name__ == "__main__":
    main()
