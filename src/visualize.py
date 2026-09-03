"""
src/visualize.py — Biểu Đồ So Sánh Kết Quả Detection
======================================================

Tạo 4 biểu đồ khoa học từ file CSV kết quả:

1. Accuracy vs Compression  — mAP50 vs Model Size
2. FPS vs Method            — Tốc độ theo từng phương pháp
3. Pruning Sweep            — mAP50 và FPS theo mức pruning
4. Summary Dashboard        — Tổng hợp đầy đủ
"""

import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, RESULTS_DIR

# Style đồng nhất
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#333344",
    "text.color": "white",
    "axes.labelcolor": "#cccccc",
    "xtick.color": "#aaaaaa",
    "ytick.color": "#aaaaaa",
    "grid.color": "#2a2d3a",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
})

# Màu sắc cho từng nhóm model
COLOR_MAP = {
    "baseline": "#4e9af1",         # Xanh dương — Baseline
    "pruned": "#f1c04e",           # Vàng — Pruned
    "finetuned": "#ff9f43",        # Cam — Pruned + Finetuned
    "quant_dynamic": "#a29bfe",    # Tím — Dynamic Quant
    "quant_onnx": "#4ef18a",       # Xanh lá — ONNX INT8
    "combined": "#ff6b81",         # Hồng — Combined
}


def _get_color(model_name: str) -> str:
    name = model_name.lower()
    if "baseline" in name:
        return COLOR_MAP["baseline"]
    if "finetuned" in name or "ft" in name:
        return COLOR_MAP["finetuned"]
    if "pruned" in name:
        return COLOR_MAP["pruned"]
    if "onnx" in name and ("pruned" in name or "combined" in name):
        return COLOR_MAP["combined"]
    if "onnx" in name:
        return COLOR_MAP["quant_onnx"]
    if "dynamic" in name:
        return COLOR_MAP["quant_dynamic"]
    return "#888888"


def _short_name(name: str) -> str:
    """Tên ngắn gọn cho trục biểu đồ."""
    name = name.replace("_finetuned", "_ft").replace("pct", "%")
    name = name.replace("quant_", "").replace("_int8", " INT8")
    name = name.replace("pruned_", "P")
    return name


# =============================================================================
# 1. mAP50 VS SIZE (Accuracy - Compression tradeoff)
# =============================================================================
def plot_accuracy_vs_size(df: pd.DataFrame) -> Path:
    """Scatter plot: mAP50 (%) vs Model Size (MB). Bubble size = FPS."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")

    valid = df[df["map50"].notna() & (df["fps"] > 0)].copy()

    for _, row in valid.iterrows():
        color = _get_color(row["model"])
        fps = max(row["fps"], 1)
        bubble_size = fps * 25  # FPS cao → bubble to
        ax.scatter(row["size_mb"], row["map50"], s=bubble_size,
                   color=color, alpha=0.85, edgecolors="white", linewidth=0.8, zorder=5)
        ax.annotate(
            _short_name(row["model"]),
            (row["size_mb"], row["map50"]),
            xytext=(6, 4), textcoords="offset points",
            color="white", fontsize=8.5,
        )

    ax.set_xlabel("Model Size (MB)", fontsize=12)
    ax.set_ylabel("mAP50 (%)", fontsize=12)
    ax.set_title("Accuracy vs Model Size\n(Bubble size = FPS - to hon la nhanh hon)",
                 color="white", fontsize=13, pad=12)
    ax.grid(True, alpha=0.3)

    # Chú thích màu sắc
    legend_patches = [
        mpatches.Patch(color=COLOR_MAP["baseline"], label="Baseline"),
        mpatches.Patch(color=COLOR_MAP["pruned"], label="Pruned"),
        mpatches.Patch(color=COLOR_MAP["finetuned"], label="Pruned + Fine-tune"),
        mpatches.Patch(color=COLOR_MAP["quant_onnx"], label="ONNX INT8"),
        mpatches.Patch(color=COLOR_MAP["combined"], label="Pruned + ONNX INT8"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=9,
              facecolor="#1a1d27", edgecolor="#444444")

    out = FIGURES_DIR / "accuracy_vs_size.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  💾 {out.name}")
    return out


# =============================================================================
# 2. FPS COMPARISON BAR CHART
# =============================================================================
def plot_fps_comparison(df: pd.DataFrame) -> Path:
    """Bar chart so sánh FPS của tất cả model."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0f1117")

    valid = df[df["fps"] > 0].copy()
    valid = valid.sort_values("fps", ascending=True)

    labels = [_short_name(n) for n in valid["model"]]
    fps_vals = valid["fps"].tolist()
    colors = [_get_color(n) for n in valid["model"]]

    bars = ax.barh(labels, fps_vals, color=colors, edgecolor="none", height=0.65)

    # Line real-time threshold
    ax.axvline(x=15, color="#ff6b81", linewidth=2, linestyle="--", alpha=0.8, zorder=6)
    ax.text(15.5, -0.5, "15 FPS\n(Real-time\nthreshold)", color="#ff6b81",
            fontsize=9, va="top")

    # Giá trị trên thanh
    for bar, val in zip(bars, fps_vals):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} FPS", va="center", color="white", fontsize=9)

    ax.set_xlabel("FPS (CPU - Edge Device Simulation)", fontsize=12)
    ax.set_title("FPS Comparison - All Models on CPU\n"
                 "(Mo phong thiet bi bien khong co GPU)",
                 color="white", fontsize=13, pad=12)
    ax.grid(True, axis="x", alpha=0.3)

    out = FIGURES_DIR / "fps_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  💾 {out.name}")
    return out


# =============================================================================
# 3. PRUNING SWEEP: mAP50 vs FPS tại các mức pruning
# =============================================================================
def plot_pruning_sweep(df: pd.DataFrame) -> Path:
    """Line chart: mAP50 và FPS theo mức pruning (% weight removed)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")

    # Lọc các model pruned có fine-tune
    pruned_df = df[df["model"].str.contains("pruned") &
                   df["model"].str.contains("finetuned") &
                   df["map50"].notna()].copy()

    def extract_pct(name: str) -> int:
        import re
        match = re.search(r"(\d+)pct", name)
        return int(match.group(1)) if match else 0

    if pruned_df.empty:
        # Không có data pruning
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "Chưa có dữ liệu pruning", transform=ax.transAxes,
                    ha="center", color="white", fontsize=13)
    else:
        pruned_df["pct"] = pruned_df["model"].apply(extract_pct)
        pruned_df = pruned_df.sort_values("pct")

        baseline_row = df[df["model"] == "baseline"]
        baseline_map = baseline_row["map50"].values[0] if not baseline_row.empty else None
        baseline_fps = baseline_row["fps"].values[0] if not baseline_row.empty else None

        # mAP50
        ax1.plot(pruned_df["pct"], pruned_df["map50"], "o-",
                 color=COLOR_MAP["finetuned"], linewidth=2.5, markersize=8)
        if baseline_map is not None:
            ax1.axhline(y=baseline_map, color=COLOR_MAP["baseline"], linewidth=1.5,
                        linestyle="--", label=f"Baseline: {baseline_map:.1f}%")
        ax1.set_xlabel("Pruning Rate (%)", fontsize=12)
        ax1.set_ylabel("mAP50 (%)", fontsize=12)
        ax1.set_title("mAP50 vs Pruning Rate\n(sau Fine-tune)", color="white", fontsize=12)
        ax1.legend(fontsize=10, facecolor="#1a1d27")
        ax1.grid(True, alpha=0.3)

        # FPS
        ax2.plot(pruned_df["pct"], pruned_df["fps"], "s-",
                 color=COLOR_MAP["quant_onnx"], linewidth=2.5, markersize=8)
        if baseline_fps is not None:
            ax2.axhline(y=baseline_fps, color=COLOR_MAP["baseline"], linewidth=1.5,
                        linestyle="--", label=f"Baseline: {baseline_fps:.1f} FPS")
        ax2.axhline(y=15, color="#ff6b81", linewidth=1.5, linestyle=":",
                    label="Real-time threshold (15 FPS)")
        ax2.set_xlabel("Pruning Rate (%)", fontsize=12)
        ax2.set_ylabel("FPS (CPU)", fontsize=12)
        ax2.set_title("FPS vs Pruning Rate\n(sau Fine-tune)", color="white", fontsize=12)
        ax2.legend(fontsize=9, facecolor="#1a1d27")
        ax2.grid(True, alpha=0.3)

    fig.suptitle("Pruning Analysis - Accuracy vs Speed Tradeoff",
                 color="white", fontsize=14, fontweight="bold")

    out = FIGURES_DIR / "pruning_sweep.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  💾 {out.name}")
    return out


# =============================================================================
# 4. SUMMARY DASHBOARD
# =============================================================================
def plot_summary_dashboard(df: pd.DataFrame) -> Path:
    """Dashboard tổng hợp 3 metric chính: mAP50, FPS, Size."""
    key_models = ["baseline"]

    # Chọn model tiêu biểu nhất
    for pattern in ["pruned_40pct_finetuned", "quant_onnx_int8",
                     "pruned_40pct_finetuned_onnx_int8"]:
        if pattern in df["model"].values:
            key_models.append(pattern)

    subset = df[df["model"].isin(key_models)].copy()
    if subset.empty:
        subset = df.head(4)

    labels = [_short_name(n) for n in subset["model"]]
    colors = [_get_color(n) for n in subset["model"]]
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Model Optimization Summary Dashboard\n"
                 "Bien Bao Giao Thong VN | Quantization + Pruning Research",
                 color="white", fontsize=14, fontweight="bold")

    # Panel 1: mAP50
    ax = axes[0]
    vals = subset["map50"].fillna(0).tolist()
    bars = ax.bar(x, vals, color=colors, edgecolor="none", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("mAP50 (%)")
    ax.set_title("Accuracy (mAP50)", color="white")
    ax.set_ylim(0, max(vals) * 1.15 if vals else 100)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", color="white", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: FPS
    ax = axes[1]
    vals = subset["fps"].tolist()
    bars = ax.bar(x, vals, color=colors, edgecolor="none", width=0.6)
    ax.axhline(y=15, color="#ff6b81", linewidth=2, linestyle="--", label="15 FPS threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("FPS (CPU)")
    ax.set_title("Speed (FPS on CPU)", color="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", color="white", fontsize=9)
    ax.legend(fontsize=9, facecolor="#1a1d27")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: Size
    ax = axes[2]
    vals = subset["size_mb"].tolist()
    bars = ax.bar(x, vals, color=colors, edgecolor="none", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Model Size (MB)", color="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.1f}MB", ha="center", color="white", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    out = FIGURES_DIR / "summary_dashboard.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  💾 {out.name}")
    return out


# =============================================================================
# 5. GENERATE ALL
# =============================================================================
def generate_all_plots(csv_path: Path | None = None) -> list[Path]:
    """
    Đọc CSV và vẽ tất cả biểu đồ.

    Args:
        csv_path: Path đến CSV (mặc định: results/comparison_metrics.csv)

    Returns:
        List path các file PNG đã tạo
    """
    if csv_path is None:
        csv_path = RESULTS_DIR / "comparison_metrics.csv"

    if not csv_path.exists():
        print(f"  ⚠️  Chưa có CSV tại {csv_path}")
        print("  Chạy pipeline trước: uv run python main.py --phase run_all")
        return []

    print(f"\n🎨 Vẽ biểu đồ từ {csv_path.name}...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(csv_path)

    paths = []
    for fn in [plot_accuracy_vs_size, plot_fps_comparison,
                plot_pruning_sweep, plot_summary_dashboard]:
        try:
            path = fn(df)
            paths.append(path)
        except Exception as e:
            print(f"  ⚠️  Lỗi vẽ biểu đồ {fn.__name__}: {e}")

    print(f"\n✅ {len(paths)} biểu đồ đã lưu tại: {FIGURES_DIR}/")
    return paths
