"""Shared figure style for the 2026-10-08 questions report.

Every figs_<NN>_*.py module imports this so all figures share one palette, one type
scale and one output convention. Figures are static (print/PDF), light mode only.

Palette: the dataviz reference instance (validated all-pairs for the first three
slots on the light surface). Assign hues by role, never by rank:
  METHOD["pca"] blue, METHOD["fa"] orange, METHOD["factosig"] aqua (needs a direct
  label: below 3:1 contrast), direct genes / controls / "other" in muted gray.
Patients go on an axis, not on colour. Highlight the highest-prevalence patients
(P02, P03, P04) with annotation or a bracket, not a new hue.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]  # reports/<report>/ -> research -> mrd_stage0_2 -> scripts -> repo
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import add_study_module_paths, experiment_dir  # noqa: E402

FIG_DIR = HERE / "figures"

# Chart chrome and ink (light).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots, fixed order.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
METHOD = {
    "pca": BLUE,
    "fa": ORANGE,
    "factosig": AQUA,
    "factosig_promax": AQUA,
    "direct_gene": MUTED,
}
METHOD_LABEL = {
    "pca": "PCA",
    "fa": "FA",
    "factosig": "FactoSig",
    "factosig_promax": "FactoSig-promax",
    "direct_gene": "direct genes",
}

# Sequential (blue, light -> dark) and diverging (blue <-> gray <-> red).
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG = "#2a78d6"
DIV_MID = "#f0efec"
DIV_POS = "#e34948"

SUPPORTED = ("P01", "P02", "P03", "P04", "P06", "P07", "P09")
HIGH_PREVALENCE = ("P02", "P03", "P04")

# Text column is 6.5 in wide in report.tex.
FULL_W = 6.5
HALF_W = 3.15

LINE_W = 1.5  # ~2 px at print resolution
MARKER = 6.0  # >= 8 px diameter
RING = 1.2  # surface ring around markers


def apply() -> None:
    """Set matplotlib rcParams for the report."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 8.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.facecolor": SURFACE,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": LINE_W,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": MARKER,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
        }
    )


def save(fig, name: str) -> Path:
    """Write figures/<name>.pdf (embedded in the report) and a .png preview for review."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=170)
    plt.close(fig)
    return pdf


def exp_dir() -> Path:
    """The canonical experiment dir (resolved through SC_* variables)."""
    return experiment_dir()


def diag_dir(slug: str) -> Path:
    """EXP/analysis/scorecards/stage2_diagnostic/<slug> (slug '' gives the root)."""
    base = exp_dir() / "analysis" / "scorecards" / "stage2_diagnostic"
    return base / slug if slug else base
