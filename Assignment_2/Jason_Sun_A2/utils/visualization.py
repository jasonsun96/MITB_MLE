"""
Visualisation stage of the A2 ML pipeline.

Reads the gold monitoring table (the time series written by monitoring.py) and
renders the performance & stability charts that go into the slideument:
  - discrimination over time (ROC-AUC + KS), with train / val / OOT shading
  - Population Stability Index (PSI) over time, with the 0.1 / 0.25 thresholds
  - scoring volume with predicted vs actual default rate

Charts are written as PNGs to <gold_monitoring_dir>/plots so a backfill leaves
an up-to-date visual snapshot of model health on disk.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

NAVY="#1F3A5F"; BLUE="#2E6FB7"; TEAL="#1B998B"; AMBER="#E8A33D"; RED="#C8553D"; GREY="#8A94A6"

try:
    from utils.training import TRAIN_CUTOFF, VAL_CUTOFF
except Exception:
    TRAIN_CUTOFF, VAL_CUTOFF = "2023-09-01", "2023-12-01"


def _load(gold_monitoring_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(gold_monitoring_dir, "gold_monitoring_*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df.sort_values("snapshot_date")


def _xfmt(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))


def make_monitoring_charts(gold_monitoring_dir: str, out_dir: str = None):
    m = _load(gold_monitoring_dir)
    if m.empty:
        print("[visualisation] no monitoring partitions yet - skipping")
        return None
    out_dir = out_dir or os.path.join(gold_monitoring_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.edgecolor":"#cfd6e0","axes.grid":True,
        "grid.color":"#eef1f5","axes.axisbelow":True,"axes.spines.top":False,
        "axes.spines.right":False,"figure.dpi":140})
    te, ve = pd.to_datetime(TRAIN_CUTOFF), pd.to_datetime(VAL_CUTOFF)

    # discrimination
    fig, ax = plt.subplots(figsize=(9,4.2))
    ax.axvspan(m.snapshot_date.min(), te, color=BLUE, alpha=0.06)
    ax.axvspan(te, ve, color=AMBER, alpha=0.08)
    ax.axvspan(ve, m.snapshot_date.max(), color=TEAL, alpha=0.06)
    ax.plot(m.snapshot_date, m.auc, marker="o", lw=2.2, color=NAVY, label="ROC-AUC")
    ax.plot(m.snapshot_date, m.ks, marker="s", lw=1.8, color=TEAL, label="KS")
    ax.axhline(0.5, ls="--", lw=1, color=GREY); ax.set_ylim(0.4,1.03)
    ax.set_title("Model discrimination over time", fontweight="bold", color=NAVY, loc="left")
    ax.legend(frameon=False, ncol=2, loc="lower left"); _xfmt(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"discrimination_over_time.png")); plt.close(fig)

    # PSI
    fig, ax = plt.subplots(figsize=(9,4.2))
    ax.axhspan(0,0.1,color=TEAL,alpha=0.10); ax.axhspan(0.1,0.25,color=AMBER,alpha=0.12)
    ax.axhspan(0.25,max(1.4,m.psi_score.max()*1.1),color=RED,alpha=0.08)
    ax.plot(m.snapshot_date, m.psi_score, marker="o", lw=2.2, color=NAVY, label="PSI (output scores)")
    if "csi_max" in m.columns and m["csi_max"].notna().any():
        ax.plot(m.snapshot_date, m.csi_max, marker="s", lw=1.8, color=AMBER, label="CSI (max feature)")
        ax.legend(frameon=False, loc="upper left")
    ax.axhline(0.1, ls="--", lw=1, color=TEAL); ax.axhline(0.25, ls="--", lw=1, color=RED)
    ymax=max(1.4, float(np.nanmax([m.psi_score.max(), m.get("csi_max", pd.Series([0])).max()]))*1.1)
    ax.set_ylim(0, ymax); ax.set_ylabel("PSI / CSI")
    ax.set_title("Score (PSI) & feature (CSI) stability over time", fontweight="bold", color=NAVY, loc="left"); _xfmt(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"psi_stability_over_time.png")); plt.close(fig)

    # volume + rates
    fig, ax = plt.subplots(figsize=(9,4.2))
    ax.bar(m.snapshot_date, m.n_scored, width=20, color="#dce4ef", label="Loans scored")
    ax.set_ylabel("Loans scored", color=GREY); ax.set_ylim(0, m.n_scored.max()*1.7)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(m.snapshot_date, m.actual_default_rate, marker="o", lw=2.2, color=RED, label="Actual default rate")
    ax2.plot(m.snapshot_date, m.pred_positive_rate, marker="s", lw=1.8, color=BLUE, label="Predicted positive rate")
    ax2.set_ylabel("Rate"); ax2.set_ylim(0,0.45); _xfmt(ax)
    ax.set_title("Volume & predicted vs actual default rate", fontweight="bold", color=NAVY, loc="left")
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, loc="upper left", frameon=False, ncol=3, fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"volume_and_rates.png")); plt.close(fig)

    print(f"[visualisation] wrote 3 charts to {out_dir}")
    return out_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gold-monitoring-dir", default="datamart/gold/monitoring")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    make_monitoring_charts(args.gold_monitoring_dir, args.out_dir)
