"""v3 of the Pavan protein update: explainers for the user's questions on report Figure 19, Table 3 and
the old Figure 20 (asks A-20261010-14..20), extended on 2026-10-10 for asks A-20261010-21..24: PC8's other
strong malignant correlates (CD116, CD33, CD146), what the sign of rho means (panel D of the schematic),
and the same figures inside the questions report (research/reports/2026-10-08_questions_report).

In the questions report (full report, section 5) Figure 19 was fig05c_celltype, Table 3 tab:05adt and
Figure 20 fig05e_rho_anchors, all made by figs_05_pc8.py. From this version on the full report shows the
relabelled Figure 19 (fig05g), Table 3 v3 (typed in sections/05_what_is_pc8.tex from table3_*_v3.csv), the
schematic as Figure 20 (fig05h + fig05i), fig05e as Figure 21 and the two checks as Figure 22 (fig05j). The
core report (report_core.pdf, the one PDF git tracks) shows compact versions: page 7, Figure 6 bottom =
fig05m_celltype_core (this script); page 8, Table 1 = the PC8 rows of Table 3 (typed in
sections_core/06_pc8_protein.tex) and Figure 7 = fig05l (figs_05_pc8.py: panel D of the schematic, compact, and
the two checks, drawn with this script's loaders and fig6_panel). A three-page core appendix with fig05k, fig05j
and fig05g was drafted (sections_core/11_appendix_protein_figures.tex) and is not built (the core's 12-page limit).
REPORT_PLACE below prints these report numbers on the update figures, because the working names F5-F7 are not
report numbers (A-20261010-22); keep it in step with the report.
PC8 = IFN-alpha PCA K=40 factor 8, fit on RNA only and oriented with the D18 sign (x -1) so that
positive = the malignant-leaning end. FA7 = IFN-alpha FA K=40 factor 7 (D18 sign +1).

Working names (files in this directory) and what they show:
  F5  fig5_factor_protein_to_rho_v3   From a factor and a protein to one number. (A) The RNA matrix
      (cells x genes) gives the score matrix; its PC8 column, kept for the 1,569 malignant cells, is a
      vector of 1,569 values. The ADT matrix (cells x 170) gives the CLR matrix; its CD99 column is a
      second vector of 1,569 values. Spearman rho = the Pearson correlation of the two rank vectors.
      (B) A worked example with 8 real malignant cells of P03, predicted HSC/MPP2. (C) The same number at
      three levels: all 1,569 cells pooled; within one patient (median over the 7); within one patient and
      cell type (median over the 11 groups of >= 30 cells). (D) What the sign means: the two rank vectors
      of all 1,569 malignant cells plotted against each other for CD99 (rho > 0, the rankings run the same
      way) and CD48 (rho < 0, they run in reverse), with the mean protein rank per tenth of the PC8 ranking.
  F6  fig6_two_checks_v3   Per protein: x = median within-patient rho, y = median within patient-and-
      cell-type rho; (A) malignant, (B) healthy cells, PC8. Isotype references as gray bands (x: within
      patient; y: within patient and cell type), the robust rule's size bar (|x| >= 0.20) dotted, y = x
      dashed. Labels include CD116, CD33 and CD146 (A-20261010-21).
  F7  fig7_celltype_means_v3   Figure 19 relabelled: (n malignant / n healthy) per predicted type, the
      definition of the mean oriented PC8 score, and which patient the pooled malignant cells come from.
  T3  table3_rho_three_levels_v3.{md,csv}   Table 3 with plain column headers (all cells pooled; median
      within patient (patients agreeing); median within patient and cell type (groups agreeing)), the
      isotype reference for each column, plus the FA7 rows of the report v1 table (CD112, GPR56, CD63) and
      six more PC8 malignant rows (CD116, CD33, CD146 asked for; CD63, CD82, GPR56 labelled in F6).

Report versions (no title, subtitle, notes or footer: the LaTeX caption carries them; written to the
report's figures/ directory with the report's .pdf + .png convention):
  fig05g_celltype_counts       = F7 plot        (full report Figure 19)
  fig05h_rho_schematic_ab      = F5 panels A, B (full report Figure 20)
  fig05i_rho_schematic_cd      = F5 panels C, D (full report Figure 20, continued)
  fig05j_two_checks            = F6 panels      (full report Figure 22)
  fig05k_rho_schematic_core    = F5 panels A, D, lettered A, B (the unbuilt core appendix)
  fig05m_celltype_core         = F7 plot, compact: one column per type (core report Figure 6, bottom)
The report runs this script through figs_05p_protein_explainer.py (make_figures.py section "05p").

Inputs (EXP-relative, read only):
  analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/ (D27): d27_ifna_pc8_fa7_adt.csv,
    d27_card_summary.csv, d27_strata.csv, d27_celltype_means.csv
  analysis/scorecards/stage2_diagnostic/factor_cards_v1/factor_cards_celltype.csv (D18; Figure 19's source)
  stage1_dr/panel_id=single_geneset__hallmark_interferon_alpha_response/method={pca,fa}/k=40/seed=42/
    scores.npy and metadata.json
  the cohort cache (study_config cohort_cache_relpath): obs patient, CN.label, predicted.annotation;
    obsm['ADT'] raw counts (per-cell CLR as in D27). The raw AnnData is never loaded (AGENTS R8).
Every per-cell recomputation is asserted against D27 (or D18 for Figure 19) before plotting.

Outputs (this directory): fig5_*_v3, fig6_*_v3, fig7_*_v3 as .png (200 dpi), .pdf and .csv (the numbers
behind each figure); table3_rho_three_levels_v3.md and .csv. Captions: caption_v3.md (hand-written).
Report outputs: <report>/figures/fig05{g,h,i,j,k,m}_*.pdf and .png (170 dpi previews; gitignored, like every
report figure).

Run (lab env, from the repo root; about 20 s):
  python scripts/mrd_stage0_2/research/team_updates/2026-10-09_protein_association_for_pavan/make_figures_v3.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")  # AGENTS R1: BLAS threads capped at 4 in an agent shell

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]  # team_updates/<dir>/ -> research -> mrd_stage0_2 -> scripts -> repo
STYLE_DIR = REPO / "scripts" / "mrd_stage0_2" / "research" / "reports" / "2026-10-08_questions_report"
sys.path.insert(0, str(STYLE_DIR))

import report_style as rs  # noqa: E402  (sets the Agg backend and puts REPO/src on sys.path)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from scipy.stats import pearsonr, rankdata, spearmanr  # noqa: E402

from sc_classification.utils.paths import study_config  # noqa: E402

D27_SLUG = "factor_modality_assoc_v1"
D18_SLUG = "factor_cards_v1"
PANEL = "single_geneset__hallmark_interferon_alpha_response"
PC8_CARD = f"{PANEL}__pca__k40__f008"
FA7_CARD = f"{PANEL}__fa__k40__f007"
FACTORS = {"PC8": (PC8_CARD, "pca", 7), "FA7": (FA7_CARD, "fa", 6)}  # name -> (card, method, 0-based column)
SUPPORTED = list(rs.SUPPORTED)
LABELS = ("normal", "malignant")  # D27 label names; "normal" is called "healthy" in every output
SHOW = {"normal": "healthy", "malignant": "malignant"}
PCT_MIN = 30  # D27 PCT_MIN_CELLS: supported patient x predicted cell type groups
CT_MIN_POOLED = 20  # D18/D27: pooled cell-type means need >= 20 cells of the label (Figure 19's rows)
ROBUST_ABS = 0.20  # Figure 20's rule: |median within-patient rho| >= 0.20 ...
ROBUST_SAME_SIGN = 5  # ... and the same sign in >= 5 of the 7 patients
EXAMPLE_PATIENT, EXAMPLE_TYPE, EXAMPLE_PROTEIN = "P03", "HSC/MPP2", "CD99"
SIGN_EXAMPLES = ("CD99", "CD48")  # panel D: one positive and one negative pooled malignant-cell rho
N_EXAMPLE = 8
N_BINS = 10  # panel D: mean protein rank in each tenth of the PC8 ranking
OTHER_STRONG = ("CD116", "CD33", "CD146")  # PC8's other strong malignant correlates (A-20261010-21)
LABELLED = {  # Figure 6 labels (the user's requests)
    "malignant": ["CD99", "CD48", "CD63", "CD82", "GPR56", "CD146", "CD116", "CD33"],
    "normal": ["CD34", "CD22", "CD112", "CD48", "CD81"],
}
# (factor, label, protein, row source): "v2" = in the report's Table 3 before 2026-10-10; "v1" = in the
# report v1 table (top 3 by |pooled rho|); "new" = added 2026-10-10 (A-20261010-21, and the proteins Figure 6
# labels for PC8 in malignant cells).
T3_ROWS = [
    ("PC8", "normal", "CD34", "v2"), ("PC8", "normal", "CD22", "v2"),
    ("PC8", "malignant", "CD99", "v2"), ("PC8", "malignant", "CD48", "v2"),
    ("PC8", "malignant", "CD116", "new"), ("PC8", "malignant", "CD33", "new"), ("PC8", "malignant", "CD146", "new"),
    ("PC8", "malignant", "CD63", "new"), ("PC8", "malignant", "CD82", "new"), ("PC8", "malignant", "GPR56", "new"),
    ("FA7", "normal", "CD22", "v2"), ("FA7", "normal", "CD34", "v2"), ("FA7", "normal", "CD112", "v1"),
    ("FA7", "malignant", "CD116", "v2"), ("FA7", "malignant", "CLEC12A", "v2"),
    ("FA7", "malignant", "GPR56", "v1"), ("FA7", "malignant", "CD63", "v1"),
]
ROW_MARK = {"v2": "", "v1": " †", "new": " ‡"}
REPORT_FIG_DIR = rs.FIG_DIR  # the questions report's figures/ (gitignored outputs)
# Where each update figure sits in the questions report (version 3.1), printed on the update figures so that the
# working names F5-F7 are never mistaken for report numbers (A-20261010-22). Must match report.tex and
# report_core.tex: check the built PDFs after any change there.
REPORT_PLACE = {
    "F5": "In the questions report: full report Figure 20 (A-D); core report Figure 7A (panel D, compact).",
    "F6": "In the questions report: full report Figure 22; core report Figure 7B-C.",
    "F7": "In the questions report: full report Figure 19; core report Figure 6 (bottom, compact).",
    "T3": "In the questions report: full report Table 3 (all rows); core report Table 1 (the PC8 rows).",
}

FS = 8.4
FS_SMALL = 7.6
FS_NOTE = 7.4
MINUS = "−"
FACTOR_C = rs.BLUE  # PC8 (PCA colour in the report)
PROTEIN_C = rs.AQUA  # CD99 (always with a dark text label: aqua is below 3:1 on the surface)
ISOTYPE_C = rs.ORANGE  # isotype controls (as in v2 Figure 4)
BAND = "#ebeae5"
BAND_BOTH = "#d9d8d0"
MATRIX_FILL = "#f1f0ec"
MATRIX_LINE = "#dcdbd3"
LIGHT_DOT = "#c9c8bf"


def signed(v: float, nd: int = 2) -> str:
    s = f"{v:+.{nd}f}"
    if s in (f"+{0:.{nd}f}", f"-{0:.{nd}f}"):
        s = f"{0:.{nd}f}"
    return s.replace("-", MINUS)


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------

def load_d27() -> dict:
    d = rs.diag_dir(D27_SLUG)
    adt = pd.read_csv(d / "d27_ifna_pc8_fa7_adt.csv")
    adt["factor"] = adt["card_id"].map({v[0]: k for k, v in FACTORS.items()})
    assert adt["factor"].notna().all()
    assert adt.groupby(["factor", "label"]).size().eq(170).all()
    assert adt[adt["is_isotype"]].groupby(["factor", "label"]).size().eq(9).all()
    cs = pd.read_csv(d / "d27_card_summary.csv").set_index("card_id")
    orient = {name: int(cs.loc[card, "orientation_sign"]) for name, (card, _, _) in FACTORS.items()}
    assert orient == {"PC8": -1, "FA7": 1}, orient  # D18 orientation, as in the report
    st = pd.read_csv(d / "d27_strata.csv")
    pct = st[st["kind"] == "patient_celltype"][["label", "patient", "cell_type", "n_cells"]].copy()
    assert pct.groupby("label").size().to_dict() == {"malignant": 11, "normal": 71}
    pooled_n = st[st["kind"] == "pooled"].set_index("label")["n_cells"].astype(int).to_dict()
    patient_n = (st[(st["kind"] == "patient") & st["supported_patient"].astype(bool)]
                 .pivot_table(index="patient", columns="label", values="n_cells").astype(int))
    means = pd.read_csv(d / "d27_celltype_means.csv")
    means = means[means["card_id"] == PC8_CARD].copy()
    return {"adt": adt, "cs": cs, "orient": orient, "pct": pct, "pooled_n": pooled_n, "patient_n": patient_n,
            "means": means}


def isotype_refs(adt: pd.DataFrame) -> pd.DataFrame:
    """Per factor x label: the largest |rho| any of the 9 isotypes reaches at each level, and which one."""
    rows = []
    for (f, lab), g in adt[adt["is_isotype"]].groupby(["factor", "label"]):
        row = {"factor": f, "label": lab}
        for level, col in (("pooled", "rho_pooled"), ("within_patient", "rho_within_median"),
                           ("within_type", "rho_pct_median")):
            a = g[col].abs()
            row[f"iso_{level}"] = float(a.max())
            row[f"iso_{level}_set_by"] = g.loc[a.idxmax(), "adt_marker_short"].replace("Isotype-", "")
        rows.append(row)
    return pd.DataFrame(rows).set_index(["factor", "label"])


def load_cells(orient: dict) -> dict:
    """Per-cell table in cohort-cache row order, with oriented cohort-z PC8 and FA7 and per-cell CLR ADT."""
    import anndata as ad

    exp = rs.exp_dir()
    cfg = study_config()
    adata = ad.read_h5ad(exp / cfg["cohort_cache_relpath"], backed="r")
    obs = adata.obs
    assert adata.n_obs == int(cfg["cohort"]["n_cells"])
    cells = pd.DataFrame({
        "cell_id": np.asarray(adata.obs_names).astype(str),
        "patient": obs["patient"].astype(str).to_numpy(),
        "label": np.where(obs["CN.label"].astype(str).eq("cancer"), "malignant", "normal"),
        "cell_type": obs["predicted.annotation"].astype(str).to_numpy(),
    })
    adt = adata.obsm["ADT"]
    names = [str(c) for c in adt.columns]
    counts = np.asarray(adt.to_numpy(), dtype=np.float64)
    adata.file.close()
    logged = np.log1p(counts)
    clr = logged - logged.mean(axis=1, keepdims=True)  # = factor_cards.clr_per_cell (D27)
    meta = {}
    for name, (_, method, col) in FACTORS.items():
        base = exp / "stage1_dr" / f"panel_id={PANEL}" / f"method={method}" / "k=40" / "seed=42"
        x = np.asarray(np.load(base / "scores.npy"), dtype=np.float64)[:, col]
        cells[name] = orient[name] * (x - x.mean()) / x.std()  # = D27 oriented cohort z
        meta[name] = json.loads((base / "metadata.json").read_text())
    assert meta["PC8"]["n_genes"] == 92 and meta["PC8"]["n_cells"] == len(cells)
    return {"cells": cells, "clr": clr, "names": names, "counts": counts, "n_genes": int(meta["PC8"]["n_genes"]),
            "standardization": meta["PC8"]["standardization"]}


def group_rhos(cohort: dict, pct: pd.DataFrame, factor: str, label: str, protein: str) -> pd.DataFrame:
    """Spearman rho inside every D27 supported patient x cell type group (>= 30 cells) of one label."""
    cells, clr = cohort["cells"], cohort["clr"]
    j = cohort["names"].index(f"ADT-{protein}")
    out = []
    for _, g in pct[pct["label"] == label].iterrows():
        sel = ((cells["label"] == label) & (cells["patient"] == g["patient"])
               & (cells["cell_type"] == g["cell_type"])).to_numpy()
        assert sel.sum() == g["n_cells"] >= PCT_MIN, (g.to_dict(), sel.sum())
        out.append({"patient": g["patient"], "cell_type": g["cell_type"], "n_cells": int(sel.sum()),
                    "rho": float(spearmanr(cells.loc[sel, factor], clr[sel, j])[0])})
    return pd.DataFrame(out)


def patient_rhos(cohort: dict, factor: str, label: str, protein: str) -> pd.DataFrame:
    cells, clr = cohort["cells"], cohort["clr"]
    j = cohort["names"].index(f"ADT-{protein}")
    out = []
    for p in SUPPORTED:
        sel = ((cells["label"] == label) & (cells["patient"] == p)).to_numpy()
        out.append({"patient": p, "n_cells": int(sel.sum()),
                    "rho": float(spearmanr(cells.loc[sel, factor], clr[sel, j])[0])})
    return pd.DataFrame(out)


def d27_row(adt: pd.DataFrame, factor: str, label: str, protein: str) -> pd.Series:
    r = adt[(adt["factor"] == factor) & (adt["label"] == label) & (adt["adt_marker_short"] == protein)]
    assert len(r) == 1, (factor, label, protein)
    return r.iloc[0]


# ----------------------------------------------------------------------------------------------
# Figure scaffolding (all geometry in inches from the top-left corner; as in make_figures_v2.py)
# ----------------------------------------------------------------------------------------------

OVERFLOW: list = []  # filled by Canvas.save; main() fails if any text sticks out of a figure


class Canvas:
    def __init__(self, w: float, h: float):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self._overlay = None

    def ax(self, left: float, top: float, width: float, height: float):
        return self.fig.add_axes([left / self.w, 1 - (top + height) / self.h, width / self.w, height / self.h])

    @property
    def overlay(self):
        """A frameless axes over the whole canvas with data coordinates in inches (y down)."""
        if self._overlay is None:
            ax = self.fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax.set_xlim(0, self.w)
            ax.set_ylim(self.h, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_facecolor("none")
            self._overlay = ax
        return self._overlay

    def text(self, x: float, y: float, s: str, **kw):
        kw.setdefault("ha", "left")
        kw.setdefault("va", "top")
        kw.setdefault("color", rs.INK_2)
        kw.setdefault("fontsize", FS)
        return self.fig.text(x / self.w, 1 - y / self.h, s, **kw)

    @staticmethod
    def header_height(title: str, subtitle: str, note: str | None = None) -> float:
        """The y below a header() with these texts (so a canvas can be sized before it is drawn)."""
        y = 0.1 + 0.205 * (title.count("\n") + 1) + 0.08 + 0.165 * (subtitle.count("\n") + 1) + 0.06
        if note:
            y += 0.15 * (note.count("\n") + 1) + 0.04
        return y

    def header(self, title: str, subtitle: str, note: str | None = None) -> float:
        """Title (the finding), subtitle (what one mark is), optional note; returns the y below it."""
        self.text(0.1, 0.1, title, fontsize=11.6, fontweight="bold", color=rs.INK, linespacing=1.15)
        n_title = title.count("\n") + 1
        y = 0.1 + 0.205 * n_title + 0.08
        self.text(0.1, y, subtitle, fontsize=FS + 0.4, color=rs.INK_2, linespacing=1.3)
        y += 0.165 * (subtitle.count("\n") + 1) + 0.06
        if note:
            self.text(0.1, y, note, fontsize=FS_NOTE + 0.2, color=rs.MUTED, linespacing=1.3)
            y += 0.15 * (note.count("\n") + 1) + 0.04
        assert abs(y - self.header_height(title, subtitle, note)) < 1e-9
        return y

    def panel_title(self, x: float, y: float, letter: str, text: str) -> None:
        self.text(x, y, letter, fontsize=10.5, fontweight="bold", color=rs.INK, va="baseline")
        self.text(x + 0.2, y, text, fontsize=FS + 0.6, fontweight="bold", color=rs.INK, va="baseline")

    def footer(self, s: str) -> None:
        self.text(0.1, self.h - 0.08, s, fontsize=FS_NOTE - 0.6, color=rs.MUTED, va="bottom", linespacing=1.3)

    def overflow(self) -> list:
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        fw, fh = self.fig.bbox.width, self.fig.bbox.height
        bad = []
        for t in self.fig.findobj(plt.Text):
            if not t.get_visible() or not t.get_text().strip():
                continue
            bb = t.get_window_extent(renderer)
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fw + 1 or bb.y1 > fh + 1:
                bad.append((t.get_text()[:50], round(bb.x1 / self.fig.dpi, 2), round(bb.y0 / self.fig.dpi, 2)))
        return bad

    def save(self, stem: str, report: bool = False) -> None:
        """Update figures go to this directory (PNG 200 dpi); report figures to the report's figures/
        (PNG 170 dpi preview, as report_style.save). Both keep the exact canvas size (no tight bbox)."""
        bad = self.overflow()
        if bad:
            OVERFLOW.append(f"{stem}: text outside the {self.w} x {self.h} in canvas: {bad}")
        out, dpi = (REPORT_FIG_DIR, 170) if report else (HERE, 200)
        out.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"savefig.bbox": "standard"}):
            self.fig.savefig(out / f"{stem}.png", dpi=dpi)
            self.fig.savefig(out / f"{stem}.pdf")
        plt.close(self.fig)


FOOTER = ("Source: D27 (factor_modality_assoc_v1) and D18 (factor_cards_v1); post-transplant MDS MRD CITE-seq "
          "cohort, 13 patients\n(within-patient values: the 7 with ≥10 malignant cells). PC8 = IFN-alpha PCA K=40 "
          "factor 8, fit on RNA only.\nProtein = CLR of raw ADT counts (no dsb). Descriptive.")


def clean_axes(ax, grid_x: bool = True) -> None:
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=grid_x)
    ax.tick_params(axis="y", length=0, pad=4)


def arrow(ax, x0: float, y0: float, x1: float, y1: float, color: str = rs.INK_2, lw: float = 1.1) -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=9))


def draw_matrix(ax, x: float, y: float, w: float, h: float, n_cols: int, hl_col: int | None, hl_color: str,
                n_row_lines: int = 9, col_every: int = 1, hl_min_w: float = 0.07) -> float:
    """A schematic matrix (cells down, features across) with one highlighted column; returns its x centre."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=MATRIX_FILL, edgecolor=rs.AXIS, lw=0.8, zorder=1))
    for k in range(1, n_row_lines):
        yy = y + h * k / n_row_lines
        ax.plot([x, x + w], [yy, yy], color=MATRIX_LINE, lw=0.4, zorder=2)
    for k in range(col_every, n_cols, col_every):
        xx = x + w * k / n_cols
        ax.plot([xx, xx], [y, y + h], color=MATRIX_LINE, lw=0.4, zorder=2)
    if hl_col is None:
        return x + w / 2
    cw = max(w / n_cols, hl_min_w)
    xc = x + w * (hl_col + 0.5) / n_cols
    x_hl = min(max(xc - cw / 2, x), x + w - cw)
    ax.add_patch(Rectangle((x_hl, y), cw, h, facecolor=hl_color, edgecolor="none", zorder=3))
    return x_hl + cw / 2


def draw_vector(ax, x: float, y: float, w: float, box_h: float, rows: list, color: str, fill: str,
                index_labels: bool) -> None:
    """A column vector drawn as boxes; rows are (index text, value text) or None for the ellipsis row."""
    for k, row in enumerate(rows):
        yy = y + k * box_h
        if row is None:
            ax.text(x + w / 2, yy + box_h * 0.5, "⋮", ha="center", va="center", fontsize=FS_SMALL, color=rs.INK_2)
            continue
        ax.add_patch(Rectangle((x, yy), w, box_h, facecolor=fill, edgecolor=color, lw=0.9, zorder=2))
        idx, val = row
        if index_labels:
            ax.text(x + 0.05, yy + box_h * 0.53, idx, ha="left", va="center", fontsize=FS_SMALL - 1.2,
                    color=rs.MUTED, zorder=3)
            ax.text(x + w - 0.05, yy + box_h * 0.53, val, ha="right", va="center", fontsize=FS_SMALL - 0.6,
                    color=rs.INK, zorder=3)
        else:
            ax.text(x + w / 2, yy + box_h * 0.53, val, ha="center", va="center", fontsize=FS_SMALL - 0.6,
                    color=rs.INK, zorder=3)


# ----------------------------------------------------------------------------------------------
# F5: from a factor and a protein to one number
# ----------------------------------------------------------------------------------------------

def fig5_data(d27: dict, cohort: dict) -> dict:
    cells, clr, names = cohort["cells"], cohort["clr"], cohort["names"]
    j = names.index(f"ADT-{EXAMPLE_PROTEIN}")
    mal = (cells["label"] == "malignant").to_numpy()
    n_mal = int(mal.sum())
    assert n_mal == d27["pooled_n"]["malignant"] == 1569
    pc8_vec = cells.loc[mal, "PC8"].to_numpy()
    cd99_vec = clr[mal, j]
    r_pc8, r_cd99 = rankdata(pc8_vec), rankdata(cd99_vec)  # rank 1 = lowest; average ranks for ties
    rho_pooled = float(spearmanr(pc8_vec, cd99_vec)[0])
    rho_ranks = float(pearsonr(r_pc8, r_cd99)[0])
    row = d27_row(d27["adt"], "PC8", "malignant", EXAMPLE_PROTEIN)
    assert abs(rho_pooled - row["rho_pooled"]) < 1e-5 and abs(rho_ranks - rho_pooled) < 1e-9
    per_patient = patient_rhos(cohort, "PC8", "malignant", EXAMPLE_PROTEIN)
    for _, r in per_patient.iterrows():
        assert abs(r["rho"] - row[f"rho_{r['patient']}"]) < 1e-5
        assert r["n_cells"] == d27["patient_n"].loc[r["patient"], "malignant"]
    med_p = float(per_patient["rho"].median())
    assert abs(med_p - row["rho_within_median"]) < 1e-5
    k_p = int((np.sign(per_patient["rho"]) == np.sign(med_p)).sum())
    assert k_p == row["n_patients_same_sign"]
    groups = group_rhos(cohort, d27["pct"], "PC8", "malignant", EXAMPLE_PROTEIN)
    med_g = float(groups["rho"].median())
    assert abs(med_g - row["rho_pct_median"]) < 1e-5 and len(groups) == row["n_pct_strata"] == 11
    k_g = int((np.sign(groups["rho"]) == np.sign(med_g)).sum())
    # The worked example: 8 malignant cells of one patient and type, one per evenly spaced PC8 quantile
    # (the rule looks at PC8 only, never at the protein).
    sel = np.where(mal & (cells["patient"] == EXAMPLE_PATIENT).to_numpy()
                   & (cells["cell_type"] == EXAMPLE_TYPE).to_numpy())[0]
    order = sel[np.argsort(cells["PC8"].to_numpy()[sel], kind="mergesort")]
    picks = [order[int(np.floor((k + 0.5) / N_EXAMPLE * len(order)))] for k in range(N_EXAMPLE)]
    ex = pd.DataFrame({"example_cell": np.arange(1, N_EXAMPLE + 1), "cell_id": cells["cell_id"].to_numpy()[picks],
                       "patient": EXAMPLE_PATIENT, "cell_type": EXAMPLE_TYPE,
                       "pc8_oriented_z": cells["PC8"].to_numpy()[picks], "cd99_clr": clr[picks, j],
                       "cd99_raw_count": cohort["counts"][picks, j]})
    ex["rank_pc8"] = rankdata(ex["pc8_oriented_z"]).astype(int)
    ex["rank_cd99"] = rankdata(ex["cd99_clr"])
    rho_ex = float(pearsonr(ex["rank_pc8"], ex["rank_cd99"])[0])
    assert abs(rho_ex - spearmanr(ex["pc8_oriented_z"], ex["cd99_clr"])[0]) < 1e-9
    g_ex = groups[(groups["patient"] == EXAMPLE_PATIENT) & (groups["cell_type"] == EXAMPLE_TYPE)].iloc[0]
    p_ex = per_patient[per_patient["patient"] == EXAMPLE_PATIENT].iloc[0]
    # The first three malignant cells (cohort-cache order) and the last one, for the vectors in panel A.
    head_idx = [0, 1, 2, n_mal - 1]
    vec = pd.DataFrame({"malignant_cell": [i + 1 for i in head_idx],
                        "cell_id": cells.loc[mal, "cell_id"].to_numpy()[head_idx],
                        "pc8_oriented_z": pc8_vec[head_idx], "rank_pc8": r_pc8[head_idx],
                        "cd99_clr": cd99_vec[head_idx], "rank_cd99": r_cd99[head_idx]})
    # Panel D: the two rank vectors of all malignant cells, for one positive and one negative pooled rho, with
    # the mean protein rank in each tenth of the PC8 ranking (bins of 156-157 cells, cut on PC8 rank only).
    sign = {}
    bins = np.array_split(np.argsort(r_pc8, kind="mergesort"), N_BINS)
    for prot in SIGN_EXAMPLES:
        pv = clr[mal, names.index(f"ADT-{prot}")]
        rp = rankdata(pv)
        rho = float(spearmanr(pc8_vec, pv)[0])
        assert abs(rho - d27_row(d27["adt"], "PC8", "malignant", prot)["rho_pooled"]) < 1e-5
        assert abs(float(pearsonr(r_pc8, rp)[0]) - rho) < 1e-9
        sign[prot] = {"rho": rho, "rank_pc8": r_pc8, "rank_prot": rp,
                      "bin_x": np.array([r_pc8[b].mean() for b in bins]),
                      "bin_y": np.array([rp[b].mean() for b in bins]),
                      "bin_n": np.array([len(b) for b in bins])}
    assert sign[SIGN_EXAMPLES[0]]["rho"] > 0 > sign[SIGN_EXAMPLES[1]]["rho"]
    return {"n_mal": n_mal, "n_cells": len(cells), "n_adt": len(names), "adt_col": j, "rho_pooled": rho_pooled,
            "per_patient": per_patient, "med_p": med_p, "k_p": k_p, "groups": groups, "med_g": med_g, "k_g": k_g,
            "example": ex, "rho_ex": rho_ex, "g_ex": g_ex, "p_ex": p_ex, "vec": vec,
            "group_patients": sorted(groups["patient"].unique()), "sign": sign,
            "n_mal_patients": int(cells.loc[mal, "patient"].nunique())}


FIG5_TITLE = "From a factor and a protein to one number (Spearman ρ),\nand what its sign means"
FIG5_NOTE = (REPORT_PLACE["F5"] + "\nPC8 is fit on RNA only; the surface protein (CITE-seq) is compared with it "
             "only afterwards.")
# Vertical extent of each panel, in inches from its title baseline: to the next panel's title (GAP) and to the
# bottom of its own last line (END). Used to size the canvas before drawing; the overflow check backs them up.
PANEL_GAP = {"A": 4.19, "B": 2.83, "C": 3.78, "D": 3.62}
PANEL_END = {"A": 3.95, "B": 2.47, "C": 3.47, "D": 3.38}
TOP_PAD = 0.25  # report versions: first panel title baseline


def fig5_subtitle(d: dict) -> str:
    return (f"Worked through for PC8 and {EXAMPLE_PROTEIN} in the {d['n_mal']:,} malignant cells. The same steps give "
            "every ρ in the\nreport's Table 3 and in its other surface-protein figures.")


def panel_a(c: Canvas, d: dict, cohort: dict, yA: float, letter: str) -> None:
    """Matrices to two vectors of n_mal values, ranks, and their Pearson correlation."""
    ov = c.overlay
    c.panel_title(0.1, yA, letter, f"Each malignant cell gets one PC8 score and one {EXAMPLE_PROTEIN} level: "
                                   f"two vectors of {d['n_mal']:,} numbers")
    lane_top = {"rna": yA + 0.42, "adt": yA + 2.04}
    mh = 0.95  # matrix height
    mx, mw = 0.32, 0.78
    sx, sw = 1.78, 0.78
    vx, vw = 3.28, 0.86
    rx, rw = 4.62, 0.6
    box_h = mh / 5
    vec = d["vec"]
    for lane in ("rna", "adt"):
        t = lane_top[lane]
        if lane == "rna":
            draw_matrix(ov, mx, t, mw, mh, n_cols=23, hl_col=None, hl_color=FACTOR_C)
            c.text(mx, t - 0.05, "RNA", fontsize=FS, fontweight="bold", color=rs.INK, va="bottom")
            c.text(mx, t + mh + 0.05, f"{d['n_cells']:,} cells ×\n{cohort['n_genes']} IFN-α genes",
                   fontsize=FS_SMALL - 0.6, linespacing=1.15)
            xc = draw_matrix(ov, sx, t, sw, mh, n_cols=40, hl_col=7, hl_color=FACTOR_C, col_every=4)
            c.text(sx, t - 0.05, "factor scores", fontsize=FS, fontweight="bold", color=rs.INK, va="bottom")
            c.text(sx, t + mh + 0.05, f"{d['n_cells']:,} cells ×\n40 factors", fontsize=FS_SMALL - 0.6,
                   linespacing=1.15)
            c.text(xc, t + 0.02, "PC8", fontsize=FS_SMALL - 1.0, color="white", fontweight="bold", ha="center",
                   rotation=90)
            arrow(ov, mx + mw + 0.07, t + mh / 2, sx - 0.07, t + mh / 2)
            c.text((mx + mw + sx) / 2, t + mh / 2 - 0.06, "PCA\n(K = 40)", fontsize=FS_SMALL - 0.6, ha="center",
                   va="bottom", linespacing=1.1)
            step2 = "column 8,\nmalignant\ncells only"
            vals = [(f"{int(r.malignant_cell):,}", signed(r.pc8_oriented_z)) for r in vec.itertuples()]
            ranks = [f"{int(r.rank_pc8):,}" for r in vec.itertuples()]
            color, name = FACTOR_C, "PC8 score"
        else:
            draw_matrix(ov, mx, t, mw, mh, n_cols=34, hl_col=None, hl_color=PROTEIN_C, col_every=2)
            c.text(mx, t - 0.05, "ADT counts", fontsize=FS, fontweight="bold", color=rs.INK, va="bottom")
            c.text(mx, t + mh + 0.05, f"{d['n_cells']:,} cells ×\n{d['n_adt']} antibodies", fontsize=FS_SMALL - 0.6,
                   linespacing=1.15)
            xc = draw_matrix(ov, sx, t, sw, mh, n_cols=d["n_adt"], hl_col=d["adt_col"], hl_color=PROTEIN_C,
                             col_every=10)
            c.text(sx, t - 0.05, "CLR protein", fontsize=FS, fontweight="bold", color=rs.INK, va="bottom")
            c.text(sx, t + mh + 0.05, f"{d['n_cells']:,} cells ×\n{d['n_adt']} antibodies", fontsize=FS_SMALL - 0.6,
                   linespacing=1.15)
            c.text(xc, t + 0.02, EXAMPLE_PROTEIN, fontsize=FS_SMALL - 1.0, color=rs.INK, fontweight="bold",
                   ha="center", rotation=90)
            arrow(ov, mx + mw + 0.07, t + mh / 2, sx - 0.07, t + mh / 2)
            c.text((mx + mw + sx) / 2, t + mh / 2 - 0.06, "CLR,\nper cell", fontsize=FS_SMALL - 0.6, ha="center",
                   va="bottom", linespacing=1.1)
            step2 = f"{EXAMPLE_PROTEIN} column,\nmalignant\ncells only"
            vals = [(f"{int(r.malignant_cell):,}", signed(r.cd99_clr)) for r in vec.itertuples()]
            ranks = [f"{r.rank_cd99:,.0f}" for r in vec.itertuples()]
            color, name = PROTEIN_C, f"{EXAMPLE_PROTEIN} level"
        arrow(ov, sx + sw + 0.07, t + mh / 2, vx - 0.07, t + mh / 2)
        c.text((sx + sw + vx) / 2, t + mh / 2 - 0.06, step2, fontsize=FS_SMALL - 0.6, ha="center", va="bottom",
               linespacing=1.1)
        rows = vals[:3] + [None] + vals[3:]
        fill = "#e3eefb" if lane == "rna" else "#dff3ea"
        draw_vector(ov, vx, t, vw, box_h, rows, color, fill, index_labels=True)
        c.text(vx, t - 0.05, name, fontsize=FS, fontweight="bold", color=rs.INK, va="bottom")
        c.text(vx, t + mh + 0.05, f"vector of {d['n_mal']:,}\n(cell 1 … {d['n_mal']:,})", fontsize=FS_SMALL - 0.6,
               linespacing=1.15)
        arrow(ov, vx + vw + 0.06, t + mh / 2, rx - 0.06, t + mh / 2)
        c.text((vx + vw + rx) / 2, t + mh / 2 - 0.06, "rank", fontsize=FS_SMALL - 0.6, ha="center", va="bottom")
        rrows = [("", v) for v in ranks[:3]] + [None] + [("", ranks[3])]
        draw_vector(ov, rx, t, rw, box_h, rrows, color, rs.SURFACE, index_labels=False)
        c.text(rx, t - 0.05, "rank", fontsize=FS, fontweight="bold", color=rs.INK, va="bottom")
        c.text(rx, t + mh + 0.05, f"1 … {d['n_mal']:,}", fontsize=FS_SMALL - 0.6)
    # converge on the result
    bx, bw = 5.55, 1.37
    y_mid_rna = lane_top["rna"] + mh / 2
    y_mid_adt = lane_top["adt"] + mh / 2
    by, bh = (y_mid_rna + y_mid_adt) / 2 - 0.5, 1.0
    for ym in (y_mid_rna, y_mid_adt):
        target = by + 0.3 if ym < by else by + bh - 0.3
        arrow(ov, rx + rw + 0.05, ym, bx - 0.04, target)
    ov.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#f4f3ef", edgecolor=rs.INK_2, lw=0.9))
    c.text(bx + bw / 2, by + 0.1, "Pearson correlation\nof the two rank\nvectors", fontsize=FS_SMALL - 0.2,
           ha="center", linespacing=1.15, color=rs.INK_2)
    c.text(bx + bw / 2, by + 0.6, f"= Spearman ρ\n= {signed(d['rho_pooled'], 3)}", fontsize=FS + 0.4,
           ha="center", fontweight="bold", color=rs.INK, linespacing=1.15)
    c.text(0.32, lane_top["adt"] + mh + 0.48,
           "PC8 score = column 8 × (−1), so that positive points to the malignant-leaning end, then z-scored over "
           f"all {d['n_cells']:,} cells\n(of these steps only × (−1) changes the ranks). CLR = log(1 + count) minus "
           f"the cell's mean log(1 + count) over all {d['n_adt']}\nantibodies. Rank 1 = the lowest value. Both "
           f"vectors list the same {d['n_mal']:,} cells in the same order (cell 1 is the same cell in both).",
           fontsize=FS_NOTE - 0.4, color=rs.MUTED, linespacing=1.3)


def panel_b(c: Canvas, d: dict, yB: float, letter: str) -> None:
    """The arithmetic for 8 real malignant cells of one patient and type."""
    ov = c.overlay
    ex = d["example"]
    c.panel_title(0.1, yB, letter, f"Worked example with {N_EXAMPLE} real malignant cells ({EXAMPLE_TYPE} of "
                                   f"patient {EXAMPLE_PATIENT})")
    t0 = yB + 0.28
    cols = [(0.55, "cell", rs.INK_2), (1.45, "PC8 score", FACTOR_C), (2.2, "rank", FACTOR_C),
            (3.05, f"{EXAMPLE_PROTEIN} (CLR)", PROTEIN_C), (3.8, "rank", PROTEIN_C)]
    for x, lab, col in cols:
        c.text(x, t0, lab, fontsize=FS_SMALL, fontweight="bold", color=rs.INK, ha="center")
        ov.plot([x - 0.3, x + 0.3], [t0 + 0.2, t0 + 0.2], color=col, lw=2.0, solid_capstyle="butt")
    row_h = 0.185
    for k, r in enumerate(ex.itertuples()):
        y = t0 + 0.27 + k * row_h
        if k % 2 == 0:
            ov.add_patch(Rectangle((0.2, y - 0.02), 3.95, row_h, facecolor="#f4f3ef", edgecolor="none", zorder=0))
        vals = [str(r.example_cell), signed(r.pc8_oriented_z), f"{r.rank_pc8}", signed(r.cd99_clr),
                f"{r.rank_cd99:.0f}"]
        for (x, _, _), v in zip(cols, vals):
            c.text(x, y, v, fontsize=FS_SMALL, color=rs.INK, ha="center")
    axS = c.ax(4.62, t0 + 0.05, 1.5, 1.5)
    axS.plot([0.5, N_EXAMPLE + 0.5], [0.5, N_EXAMPLE + 0.5], color=rs.AXIS, lw=0.9, ls=(0, (3, 2)), zorder=1)
    axS.scatter(ex["rank_pc8"], ex["rank_cd99"], s=34, color=rs.INK_2, edgecolor=rs.SURFACE, linewidth=0.8, zorder=3)
    for r in ex.itertuples():
        axS.annotate(str(r.example_cell), (r.rank_pc8, r.rank_cd99), xytext=(4, 3), textcoords="offset points",
                     fontsize=FS_SMALL - 1.0, color=rs.MUTED, ha="left", va="bottom")
    axS.set_xlim(0.3, N_EXAMPLE + 0.7)
    axS.set_ylim(0.3, N_EXAMPLE + 0.7)
    axS.set_xticks(range(1, N_EXAMPLE + 1))
    axS.set_yticks(range(1, N_EXAMPLE + 1))
    axS.tick_params(labelsize=FS_SMALL - 1.0, length=2, pad=1.5)
    axS.set_xlabel("PC8 rank", fontsize=FS_SMALL - 0.4, color=rs.INK_2, labelpad=2)
    axS.set_ylabel(f"{EXAMPLE_PROTEIN} rank", fontsize=FS_SMALL - 0.4, color=rs.INK_2, labelpad=2)
    axS.grid(axis="both", visible=True)
    axS.set_aspect("equal")
    c.text(6.25, t0 + 0.12, "Pearson r of\nthe two rank\ncolumns", fontsize=FS_SMALL - 0.4, linespacing=1.15)
    c.text(6.25, t0 + 0.68, f"= ρ\n= {signed(d['rho_ex'])}", fontsize=FS, fontweight="bold", color=rs.INK,
           linespacing=1.15)
    c.text(6.25, t0 + 1.12, "dashed: ranks\nagree exactly\n(ρ = +1)", fontsize=FS_SMALL - 1.0, color=rs.MUTED,
           linespacing=1.1)
    g_ex = d["g_ex"]
    c.text(0.32, t0 + 0.27 + N_EXAMPLE * row_h + 0.12,
           f"The 8 cells: one per evenly spaced PC8 quantile among {EXAMPLE_PATIENT}'s {int(g_ex['n_cells'])} "
           f"malignant {EXAMPLE_TYPE} cells (the choice looks at PC8 only).\nEight cells are far too few to estimate "
           f"ρ: all {int(g_ex['n_cells'])} cells of this group give {signed(g_ex['rho'], 3)}, and all "
           f"{d['n_mal']:,} malignant cells {signed(d['rho_pooled'], 3)}.",
           fontsize=FS_NOTE - 0.4, color=rs.MUTED, linespacing=1.3)


def panel_c(c: Canvas, d: dict, yC: float, letter: str) -> None:
    """The same calculation pooled, within patient and within patient and cell type."""
    c.panel_title(0.1, yC, letter, "The same calculation at three levels")
    ax_top, ax_h = yC + 0.3, 2.2
    axC = c.ax(3.05, ax_top, 2.85, ax_h)
    ylim = (2.6, -0.45)
    lanes = [0.0, 1.0, 2.0]

    def lane_y(v: float) -> float:
        return ax_top + (v - ylim[1]) / (ylim[0] - ylim[1]) * ax_h

    axC.axvline(0, color=rs.INK_2, lw=0.7, zorder=1)
    for yl in (0.5, 1.5):
        axC.axhline(yl, color=rs.GRID, lw=0.8, zorder=0)
    # level 1
    axC.scatter([d["rho_pooled"]], [lanes[0]], s=46, color=rs.INK, edgecolor=rs.SURFACE, linewidth=0.9, zorder=4)
    # level 2 (vertical offsets keep close values apart; the bar is drawn behind the dots)
    pp = d["per_patient"]
    off2 = {"P02": -0.2, "P06": 0.2}
    axC.plot([d["med_p"]] * 2, [lanes[1] - 0.34, lanes[1] + 0.34], color=rs.INK, lw=1.8, zorder=2,
             solid_capstyle="butt")
    for r in pp.itertuples():
        is_ex = r.patient == EXAMPLE_PATIENT
        yy = lanes[1] + off2.get(r.patient, 0.0)
        axC.scatter([r.rho], [yy], s=46 if is_ex else 22, color=rs.INK if is_ex else rs.MUTED, edgecolor=rs.SURFACE,
                    linewidth=0.8, zorder=4 if is_ex else 3)
        if r.patient in ("P07", "P01"):
            axC.annotate(f"{r.patient} ({r.n_cells} cells)", (r.rho, yy), xytext=(0, 5), textcoords="offset points",
                         fontsize=FS_SMALL - 1.4, ha="center", va="bottom", color=rs.INK_2)
        if is_ex:
            axC.annotate(f"{r.patient} ({r.n_cells} cells)", (r.rho, yy), xytext=(r.rho + 0.05, lanes[1] + 0.4),
                         textcoords="data", fontsize=FS_SMALL - 1.2, ha="left", va="center", color=rs.INK,
                         fontweight="bold", arrowprops=dict(arrowstyle="-", color=rs.INK_2, lw=0.6, shrinkA=1,
                                                            shrinkB=3))
    # level 3
    off3 = {("P02", "LMPP"): 0.2, ("P02", "HSC/MPP1"): 0.2, ("P03", "NP1"): -0.2, ("P09", "cDC"): -0.2,
            ("P09", "HSC/MPP1"): 0.2, ("P03", "MEP-MkP"): -0.2}
    axC.plot([d["med_g"]] * 2, [lanes[2] - 0.34, lanes[2] + 0.34], color=rs.INK, lw=1.8, zorder=2,
             solid_capstyle="butt")
    for r in d["groups"].itertuples():
        is_ex = (r.patient, r.cell_type) == (EXAMPLE_PATIENT, EXAMPLE_TYPE)
        yy = lanes[2] + off3.get((r.patient, r.cell_type), 0.0)
        axC.scatter([r.rho], [yy], s=46 if is_ex else 22, color=rs.INK if is_ex else rs.MUTED,
                    edgecolor=rs.SURFACE, linewidth=0.8, zorder=4 if is_ex else 3)
        if is_ex:
            axC.annotate(f"{r.patient} · {r.cell_type} ({r.n_cells} cells)", (r.rho, yy),
                         xytext=(r.rho + 0.05, lanes[2] + 0.4), textcoords="data", fontsize=FS_SMALL - 1.2,
                         ha="left", va="center", color=rs.INK, fontweight="bold",
                         arrowprops=dict(arrowstyle="-", color=rs.INK_2, lw=0.6, shrinkA=1, shrinkB=3))
    axC.set_ylim(*ylim)
    axC.set_yticks([])
    axC.set_xlim(-0.42, 0.62)
    axC.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
    axC.set_xticklabels([signed(v, 1) if v else "0" for v in (-0.4, -0.2, 0, 0.2, 0.4, 0.6)], fontsize=FS_SMALL - 0.4)
    axC.set_xlabel(f"Spearman ρ, PC8 vs {EXAMPLE_PROTEIN} (malignant cells)", fontsize=FS_SMALL, color=rs.INK_2)
    clean_axes(axC)
    axC.spines["left"].set_visible(False)
    p_ex = d["p_ex"]
    g_ex = d["g_ex"]
    desc = [
        ("1  All malignant cells pooled",
         f"{d['n_mal']:,} cells (the {d['n_mal_patients']} patients that have any) → one ρ"),
        ("2  Within one patient",
         f"{EXAMPLE_PATIENT}: its {int(p_ex['n_cells'])} malignant cells → {signed(p_ex['rho'], 3)}.\n"
         f"Repeat in each of the {len(pp)} patients; report the\nmedian and how many share its sign"),
        ("3  Within one patient and cell type",
         f"{EXAMPLE_PATIENT} · {EXAMPLE_TYPE}: {int(g_ex['n_cells'])} cells → {signed(g_ex['rho'], 3)}.\n"
         f"Repeat in each of the {len(d['groups'])} groups with ≥{PCT_MIN} malignant\ncells "
         f"(only {', '.join(d['group_patients'])} have any); median"),
    ]
    summ = [f"ρ = {signed(d['rho_pooled'], 3)}",
            f"median {signed(d['med_p'], 3)}\n{d['k_p']} of {len(pp)} positive",
            f"median {signed(d['med_g'], 3)}\n{d['k_g']} of {len(d['groups'])} positive"]
    for yl, (head, body), s in zip(lanes, desc, summ):
        yy = lane_y(yl)
        c.text(0.3, yy - 0.2, head, fontsize=FS_SMALL, fontweight="bold", color=rs.INK)
        c.text(0.47, yy - 0.03, body, fontsize=FS_SMALL - 0.8, linespacing=1.2)
        c.text(6.0, yy - 0.1, s, fontsize=FS_SMALL - 0.2, color=rs.INK, fontweight="bold", linespacing=1.2)
    c.text(5.98, ax_top - 0.05, "bar = median;\nlarge dot = the\npanel B example", fontsize=FS_SMALL - 1.2,
           color=rs.MUTED, linespacing=1.1, va="bottom")
    c.text(0.32, ax_top + ax_h + 0.42,
           "Level 2 removes differences between patients (in score and in staining; patients are nested in "
           "batches). Level 3 also\nremoves differences between predicted cell types. The figure 'Two checks per "
           "protein' sets the level-2 and level-3\nmedians of every protein against the isotype controls.",
           fontsize=FS_NOTE - 0.4, color=rs.INK_2, linespacing=1.3)


def panel_d(c: Canvas, d: dict, yD: float, letter: str) -> None:
    """What the sign means: the two rank vectors of all malignant cells against each other, + and - examples."""
    n = d["n_mal"]
    c.panel_title(0.1, yD, letter, "What the sign means: do the two rankings run the same way, or in reverse?")
    c.text(0.3, yD + 0.1,
           f"Each gray dot is one of the {n:,} malignant cells: its PC8 rank (x) against its protein rank (y), the two "
           f"rank vectors of panel A.\nBlack line: the mean protein rank in each tenth of the PC8 ranking. Dotted: "
           f"the middle rank, where the line would stay if ρ were 0.",
           fontsize=FS_NOTE - 0.4, color=rs.INK_2, linespacing=1.3)
    top, side = yD + 0.62, 1.55
    sentences = {
        +1: ("Forward (ρ > 0)", "Cells ranked high on\nthe oriented PC8 score\n(toward the malignant-\nleaning end) "
                                "tend to\nrank high on {p} too:\nthe two rankings run\nthe same way."),
        -1: ("Reversed (ρ < 0)", "Cells ranked high on\nPC8 tend to rank low\non {p}: the two\nrankings run in\n"
                                 "opposite directions."),
    }
    ticks = [1, 500, 1000, n]
    for k, prot in enumerate(SIGN_EXAMPLES):
        s = d["sign"][prot]
        x0 = 0.6 + k * 3.4
        ax = c.ax(x0, top, side, side)
        ax.scatter(s["rank_pc8"], s["rank_prot"], s=1.4, color=LIGHT_DOT, linewidth=0, zorder=1, rasterized=True)
        ax.axhline((n + 1) / 2, color=rs.INK_2, lw=0.9, ls=(0, (1, 2)), zorder=2)
        ax.plot(s["bin_x"], s["bin_y"], color=rs.INK, lw=1.6, marker="o", ms=3.2, mec=rs.SURFACE, mew=0.6, zorder=3)
        ax.set_xlim(-30, n + 31)
        ax.set_ylim(-30, n + 31)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"{t:,}" for t in ticks], fontsize=FS_SMALL - 1.4)
        ax.set_yticklabels([f"{t:,}" for t in ticks], fontsize=FS_SMALL - 1.4)
        ax.tick_params(length=2, pad=1.5)
        ax.grid(axis="both", visible=False)
        ax.set_aspect("equal")
        ax.set_xlabel("PC8 rank (1 = lowest)", fontsize=FS_SMALL - 0.6, color=rs.INK_2, labelpad=2)
        ax.set_ylabel(f"{prot} rank", fontsize=FS_SMALL - 0.6, color=rs.INK_2, labelpad=1)
        sg = 1 if s["rho"] > 0 else -1
        head, body = sentences[sg]
        tx = x0 + side + 0.14
        c.text(tx, top - 0.02, f"{prot}: ρ = {signed(s['rho'], 3)}", fontsize=FS, fontweight="bold", color=rs.INK)
        c.text(tx, top + 0.22, head, fontsize=FS_SMALL, fontweight="bold", color=rs.INK_2)
        c.text(tx, top + 0.42, body.format(p=prot), fontsize=FS_SMALL - 0.6, color=rs.INK_2, linespacing=1.25)
    rules = (
        "• |ρ| says how consistently the two rankings agree: +1 or −1 = exactly the same or exactly the reversed "
        "order, 0 = no tendency.\n   At +0.22 and −0.19 the tendency shows only on average (the black lines), "
        "not cell by cell.\n"
        "• The sign means something only because PC8 was oriented (× −1, so that + points to the malignant-leaning "
        "end). A raw\n   PCA sign is arbitrary: flipping it would flip every ρ and change nothing else.\n"
        "• ρ is computed among the cells of one label only (here malignant). A negative ρ does not make a protein "
        "'less malignant'\n   or 'less healthy': no healthy cell enters this number."
    )
    c.text(0.3, top + side + 0.42, rules, fontsize=FS_NOTE - 0.4, color=rs.INK_2, linespacing=1.32)


PANEL_FN = {"A": lambda c, d, cohort, y, lt: panel_a(c, d, cohort, y, lt),
            "B": lambda c, d, cohort, y, lt: panel_b(c, d, y, lt),
            "C": lambda c, d, cohort, y, lt: panel_c(c, d, y, lt),
            "D": lambda c, d, cohort, y, lt: panel_d(c, d, y, lt)}


def fig5_canvas(d: dict, cohort: dict, panels: str, letters: str, header: bool) -> Canvas:
    """Draw the given panels (in order, lettered as given) on one 7 in wide canvas sized to fit them."""
    sub = fig5_subtitle(d)
    top = (Canvas.header_height(FIG5_TITLE, sub, FIG5_NOTE) + 0.2) if header else TOP_PAD
    body = sum(PANEL_GAP[p] for p in panels[:-1]) + PANEL_END[panels[-1]]
    c = Canvas(7.0, top + body + (0.62 if header else 0.08))
    if header:
        c.header(FIG5_TITLE, sub, FIG5_NOTE)
    y = top
    for p, lt in zip(panels, letters):
        PANEL_FN[p](c, d, cohort, y, lt)
        y += PANEL_GAP[p]
    if header:
        c.footer(FOOTER)
    return c


def figure5(d: dict, cohort: dict) -> None:
    fig5_canvas(d, cohort, "ABCD", "ABCD", header=True).save("fig5_factor_protein_to_rho_v3")
    fig5_canvas(d, cohort, "AB", "AB", header=False).save("fig05h_rho_schematic_ab", report=True)
    fig5_canvas(d, cohort, "CD", "CD", header=False).save("fig05i_rho_schematic_cd", report=True)
    fig5_canvas(d, cohort, "AD", "AB", header=False).save("fig05k_rho_schematic_core", report=True)

    pp = d["per_patient"]
    rows = [{"level": "1 all malignant cells pooled", "patient": "all", "cell_type": "all", "n_cells": d["n_mal"],
             "rho": d["rho_pooled"]}]
    rows += [{"level": "2 within patient", "patient": r.patient, "cell_type": "all", "n_cells": r.n_cells,
              "rho": r.rho} for r in pp.itertuples()]
    rows.append({"level": "2 within patient: median", "patient": f"{d['k_p']} of {len(pp)} same sign",
                 "cell_type": "", "n_cells": np.nan, "rho": d["med_p"]})
    rows += [{"level": "3 within patient and cell type", "patient": r.patient, "cell_type": r.cell_type,
              "n_cells": r.n_cells, "rho": r.rho} for r in d["groups"].itertuples()]
    rows.append({"level": "3 within patient and cell type: median",
                 "patient": f"{d['k_g']} of {len(d['groups'])} same sign", "cell_type": "", "n_cells": np.nan,
                 "rho": d["med_g"]})
    out = pd.DataFrame(rows)
    out["protein"] = EXAMPLE_PROTEIN
    out["factor"] = "PC8"
    out["cells"] = "malignant"
    ex_out = d["example"].assign(level=f"B worked example ({N_EXAMPLE} cells)", rho_8_cells=d["rho_ex"])
    vec_out = d["vec"].assign(level="A first three and last malignant cells (cohort-cache order)")
    sign_out = pd.concat([pd.DataFrame({
        "level": f"D sign example: mean {prot} rank per tenth of the PC8 ranking", "protein": prot, "factor": "PC8",
        "cells": "malignant", "patient": "all", "rho": d["sign"][prot]["rho"], "bin": np.arange(1, N_BINS + 1),
        "n_cells": d["sign"][prot]["bin_n"], "mean_rank_pc8": d["sign"][prot]["bin_x"],
        "mean_rank_protein": d["sign"][prot]["bin_y"]}) for prot in SIGN_EXAMPLES], ignore_index=True)
    pd.concat([out, ex_out, vec_out, sign_out], ignore_index=True).round(4).to_csv(
        HERE / "fig5_factor_protein_to_rho_v3.csv", index=False)


# ----------------------------------------------------------------------------------------------
# F6: two checks per protein
# ----------------------------------------------------------------------------------------------

LABEL_OFFSETS = {  # (dx, dy) in points, placed by eye
    ("malignant", "CD99"): (-6, 6), ("malignant", "CD63"): (-6, 0),
    ("malignant", "CD82"): (6, 3), ("malignant", "GPR56"): (6, 0), ("malignant", "CD146"): (5, 7),
    ("malignant", "CD116"): (5, -8),
    ("normal", "CD34"): (5, 8), ("normal", "CD22"): (-6, 7), ("normal", "CD112"): (6, -8),
    ("normal", "CD48"): (-6, -8), ("normal", "CD81"): (6, -8),
}
LABEL_AT = {  # crowded corner (CD48, CD116, CD33 within 0.04 of each other): label at (x, y) in data units + leader
    ("malignant", "CD48"): (-0.348, 0.098),
    ("malignant", "CD33"): (-0.245, 0.068),
}


def fig6_panel(ax, adt: pd.DataFrame, iso: pd.Series, lab: str, key: bool = True) -> pd.DataFrame:
    a = adt[(adt["factor"] == "PC8") & (adt["label"] == lab)]
    prot = a[~a["is_isotype"]].copy()
    isos = a[a["is_isotype"]].copy()
    bx, by = float(iso["iso_within_patient"]), float(iso["iso_within_type"])
    lim = 0.36
    ax.axvspan(-bx, bx, color=BAND, lw=0, zorder=0)
    ax.axhspan(-by, by, color=BAND, lw=0, zorder=0)
    ax.add_patch(Rectangle((-bx, -by), 2 * bx, 2 * by, facecolor=BAND_BOTH, edgecolor="none", zorder=0))
    ax.plot([-lim, lim], [-lim, lim], color=rs.MUTED, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
    ax.axhline(0, color=rs.INK_2, lw=0.6, zorder=1)
    ax.axvline(0, color=rs.INK_2, lw=0.4, zorder=1)
    for xv in (-ROBUST_ABS, ROBUST_ABS):
        ax.axvline(xv, color=rs.INK_2, lw=0.7, ls=(0, (1, 2)), zorder=1)
    named = prot["adt_marker_short"].isin(LABELLED[lab])
    ax.scatter(prot.loc[~named, "rho_within_median"], prot.loc[~named, "rho_pct_median"], s=11, color=LIGHT_DOT,
               edgecolor="none", zorder=2)
    ax.scatter(isos["rho_within_median"], isos["rho_pct_median"], s=18, color=ISOTYPE_C, edgecolor=rs.SURFACE,
               linewidth=0.6, zorder=3, marker="D")
    focus = {"malignant": ("CD99", "CD48"), "normal": ("CD34",)}[lab]
    for _, r in prot[named].iterrows():
        mk = r["adt_marker_short"]
        big = mk in focus
        ax.scatter([r["rho_within_median"]], [r["rho_pct_median"]], s=34 if big else 24, color=rs.INK if big else rs.INK_2,
                   edgecolor=rs.SURFACE, linewidth=0.8, zorder=4)
        style = dict(fontsize=FS_SMALL - (0.0 if big else 0.6), color=rs.INK if big else rs.INK_2,
                     fontweight="bold" if big else "normal", va="center", zorder=5)
        if (lab, mk) in LABEL_AT:
            ax.annotate(mk, (r["rho_within_median"], r["rho_pct_median"]), xytext=LABEL_AT[(lab, mk)],
                        textcoords="data", ha="left", **style,
                        arrowprops=dict(arrowstyle="-", color=rs.INK_2, lw=0.6, shrinkA=1, shrinkB=3.5))
            continue
        dx, dy = LABEL_OFFSETS.get((lab, mk), (5, 4))
        ax.annotate(mk, (r["rho_within_median"], r["rho_pct_median"]), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", **style)
    # When one isotype sets both bands (PC8, malignant cells: RTK2071), outline its diamond and name it in the
    # corner note instead of labelling it in the crowd, where its label sat next to the CD146 dot.
    setter = iso["iso_within_patient_set_by"]
    one_setter = setter == iso["iso_within_type_set_by"]
    if one_setter:
        r = isos[isos["adt_marker_short"] == f"Isotype-{setter}"].iloc[0]
        ax.scatter([r["rho_within_median"]], [r["rho_pct_median"]], s=30, color=ISOTYPE_C, edgecolor=rs.INK,
                   linewidth=1.0, zorder=3.5, marker="D")
    # Region labels on the two reference lines, so the reading does not flip (user, 2026-10-10): a protein near
    # y = 0 tracks PC8 only because cell types differ; a protein near y = x keeps its correlation inside one type.
    # Drawn as a small key in the empty top-left corner (labels on the lines collided with CD48/CD99).
    # The compact core version (key=False) carries the same two phrases in its legend row instead.
    if key:
        region = dict(fontsize=FS_SMALL - 0.6, color=rs.INK, style="italic", va="center", zorder=5)
        ax.plot([-0.345, -0.300], [0.318, 0.318], color=rs.MUTED, lw=0.9, ls=(0, (4, 2.5)), zorder=5)
        ax.text(-0.290, 0.318, "on y = x: holds within cell type", **region)
        ax.plot([-0.345, -0.300], [0.272, 0.272], color=rs.INK_2, lw=0.8, zorder=5)
        ax.text(-0.290, 0.272, "on y ≈ 0: explained by cell type", **region)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ticks = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    lab_t = [signed(t, 1) if t else "0" for t in ticks]
    ax.set_xticklabels(lab_t, fontsize=FS_SMALL - 0.8)
    ax.set_yticklabels(lab_t, fontsize=FS_SMALL - 0.8)
    ax.grid(axis="both", visible=False)
    ax.set_xlabel("x: median within-patient ρ (7 patients)", fontsize=FS_SMALL - 0.2, color=rs.INK_2)
    ax.set_ylabel("y: median within patient-and-type ρ", fontsize=FS_SMALL - 0.2, color=rs.INK_2)
    ax.text(0.98, 0.02, f"isotype ref.: x ±{bx:.3f}, y ±{by:.3f}"
            + (f"\n(both set by {setter}, outlined)" if one_setter else ""), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=FS_SMALL - 1.2, color=rs.INK_2, linespacing=1.15)
    out = a[["adt_marker_short", "is_isotype", "rho_within_median", "n_patients_same_sign", "rho_pct_median",
             "n_pct_strata", "n_pct_same_sign"]].copy()
    out.insert(0, "cells", SHOW[lab])
    out["labelled"] = out["adt_marker_short"].isin(LABELLED[lab])
    out["outside_x_band"] = out["rho_within_median"].abs() > bx
    out["outside_y_band"] = out["rho_pct_median"].abs() > by
    out["meets_fig20_rule"] = ((out["rho_within_median"].abs() >= ROBUST_ABS)
                               & (out["n_patients_same_sign"] >= ROBUST_SAME_SIGN) & ~out["is_isotype"])
    out["iso_ref_x"] = bx
    out["iso_ref_y"] = by
    return out


FIG6_TITLE = "Two checks per protein: does it track PC8 within patients (x),\nand does that survive within cell types (y)?"
FIG6_SUB = ("One dot = one surface protein against PC8 (oriented: + = the malignant-leaning end).\n"
            "x = median of the 7 within-patient ρ; y = median of the ρ inside each patient's predicted\n"
            "cell types (groups of ≥30 cells of that label).")
FIG6_NOTE = (REPORT_PLACE["F6"] + "\nGray bands: how far the 9 isotype-control antibodies (no target) reach at "
             "each level; the darker square is inside both.")


def fig6_canvas(d27: dict, iso: pd.DataFrame, full: bool) -> tuple:
    """full = the update figure (title, reading key, footer); otherwise the report version (panels + legend)."""
    W = 7.0
    side = 2.75
    top = (Canvas.header_height(FIG6_TITLE, FIG6_SUB, FIG6_NOTE) + 0.5) if full else 0.45
    yl = top + 0.05 + side + 0.5
    H = yl + 2.9 if full else yl + 0.45
    c = Canvas(W, H)
    adt = d27["adt"]
    r99 = d27_row(adt, "PC8", "malignant", "CD99")
    r48 = d27_row(adt, "PC8", "malignant", "CD48")
    im = iso.loc[("PC8", "malignant")]
    assert np.sign(r99["rho_within_median"]) == np.sign(r99["rho_pct_median"])  # so n_pct_same_sign = own-sign count
    if full:
        c.header(FIG6_TITLE, FIG6_SUB, FIG6_NOTE)
    c.panel_title(0.1, top - 0.18, "A", "Malignant cells")
    c.text(0.3, top - 0.13, f"{d27['pooled_n']['malignant']:,} cells; {int(r99['n_pct_strata'])} type groups "
                            "from P02, P03, P04, P09", fontsize=FS_SMALL - 0.8, color=rs.INK_2)
    c.panel_title(3.6, top - 0.18, "B", "Healthy cells")
    c.text(3.8, top - 0.13, f"{d27['pooled_n']['normal']:,} cells; "
                            f"{int(d27_row(adt, 'PC8', 'normal', 'CD34')['n_pct_strata'])} type groups from 7 patients",
           fontsize=FS_SMALL - 0.8, color=rs.INK_2)
    axA = c.ax(0.62, top + 0.05, side, side)
    axB = c.ax(4.1, top + 0.05, side, side)
    tA = fig6_panel(axA, adt, im, "malignant")
    tB = fig6_panel(axB, adt, iso.loc[("PC8", "normal")], "normal")
    # legend
    handles = [Line2D([], [], ls="none", marker="o", ms=5.5, mfc=rs.INK_2, mec=rs.SURFACE),
               Line2D([], [], ls="none", marker="o", ms=4.5, mfc=LIGHT_DOT, mec="none"),
               Line2D([], [], ls="none", marker="D", ms=4.5, mfc=ISOTYPE_C, mec=rs.SURFACE),
               Line2D([], [], color=rs.MUTED, lw=0.9, ls=(0, (4, 2.5))),
               Line2D([], [], color=rs.INK_2, lw=0.8, ls=(0, (1, 2)))]
    c.fig.legend(handles, ["protein named in the text", "other protein (161 targeted in all)", "isotype control (9)",
                           "y = x: same size inside cell types",
                           f"|x| = {ROBUST_ABS:.2f}: size bar of D27's pre-set\n'robust' rule (it reads x only)"],
                 loc="upper left", bbox_to_anchor=(0.3 / W, 1 - yl / H), ncol=3, frameon=False,
                 fontsize=FS_SMALL - 0.8, handletextpad=0.4, columnspacing=1.2, labelspacing=0.5)
    if full:
        nA = tA[~tA["is_isotype"]]
        nB = tB[~tB["is_isotype"]]
        oth = [d27_row(adt, "PC8", "malignant", p) for p in OTHER_STRONG]
        key_y = yl + 0.5
        c.text(0.3, key_y, "How to read it", fontsize=FS_SMALL + 0.2, fontweight="bold", color=rs.INK)
        key = (
            "• Left or right of the vertical band: the protein tracks PC8 within patients more strongly than any "
            "isotype does.\n"
            "• Above or below the horizontal band: some of that is still there inside one patient's cell type.\n"
            "• On the dashed diagonal a correlation keeps its full size inside cell types; on y = 0 it is all a "
            "difference between types.\n"
            "• Bottom line: near y = 0 the protein mostly marks which cell types sit at which end of PC8 (cell "
            "identity); up the diagonal\n   it tracks PC8 among cells of the same type (a candidate beyond cell "
            "identity: CD99, CD63, CD82; weak, near the bands).\n"
            f"• CD48 (malignant): x {signed(r48['rho_within_median'], 3)} ({int(r48['n_patients_same_sign'])}/7) is "
            "outside the band and past the size bar (robust), but y is "
            f"{signed(r48['rho_pct_median'], 3)}: it collapses\n   within cell type.\n"
            f"• {', '.join(OTHER_STRONG[:-1])} and {OTHER_STRONG[-1]} (malignant; PC8's other strong correlates): x "
            + ", ".join(f"{signed(r['rho_within_median'], 3)} ({int(r['n_patients_same_sign'])}/7)" for r in oth)
            + ",\n   all outside the band and past the bar (robust), but y "
            + ", ".join(signed(r["rho_pct_median"], 3) for r in oth)
            + ": like CD48, near zero within cell types.\n"
            f"• CD99 (malignant): x {signed(r99['rho_within_median'], 3)} ({int(r99['n_patients_same_sign'])}/7) is "
            f"inside the band (±{im['iso_within_patient']:.3f}) and short of the bar, so the rule calls it not "
            f"robust;\n   y {signed(r99['rho_pct_median'], 3)} ({int(r99['n_pct_same_sign'])} of "
            f"{int(r99['n_pct_strata'])} groups positive) is above its band (±{im['iso_within_type']:.3f}) and near the "
            "diagonal. The rule reads x only.\n"
            f"• The rule's {ROBUST_ABS:.2f} bar lies inside the malignant x band (±{im['iso_within_patient']:.3f}): "
            f"{int((nA['meets_fig20_rule'] & ~nA['outside_x_band']).sum())} of the "
            f"{int(nA['meets_fig20_rule'].sum())} proteins that pass the rule stay inside it.\n"
            f"• Malignant: {int(nA['outside_y_band'].sum())} of {len(nA)} proteins sit outside the y band and "
            f"{int((nA['outside_x_band'] & nA['outside_y_band']).sum())} outside both bands; one isotype "
            f"({im['iso_within_patient_set_by']}) sets both bands.\n"
            f"• Healthy: {'every' if (nB['rho_pct_median'].abs() <= 0.1).all() else 'nearly every'} protein falls "
            f"toward y = 0 ({int((nB['rho_pct_median'].abs() <= 0.1).sum())} of "
            f"{len(nB)} have |y| ≤ 0.10): PC8's healthy-cell protein correlates mostly\n   say which cell types "
            "sit at which end of PC8."
        )
        c.text(0.3, key_y + 0.22, key, fontsize=FS_NOTE - 0.4, color=rs.INK_2, linespacing=1.35)
        c.footer(FOOTER)
    return c, tA, tB


def figure6(d27: dict, iso: pd.DataFrame) -> pd.DataFrame:
    im = iso.loc[("PC8", "malignant")]
    for p in OTHER_STRONG:  # the user's three (A-20261010-21): robust within patients, near zero within types
        r = d27_row(d27["adt"], "PC8", "malignant", p)
        assert abs(r["rho_within_median"]) >= ROBUST_ABS and r["n_patients_same_sign"] >= ROBUST_SAME_SIGN, p
        assert abs(r["rho_within_median"]) > im["iso_within_patient"] and abs(r["rho_pct_median"]) < 0.05, p
    c, tA, tB = fig6_canvas(d27, iso, full=True)
    nA = tA[~tA["is_isotype"]]
    passers = set(nA.loc[nA["meets_fig20_rule"], "adt_marker_short"])
    inside = set(nA.loc[nA["meets_fig20_rule"] & ~nA["outside_x_band"], "adt_marker_short"])
    assert len(passers) == 14 and inside == {"CD244", "CD11c", "CD112", "CD162", "CD7"}, (passers, inside)
    assert set(nA.loc[nA["outside_x_band"], "adt_marker_short"]) == passers - inside  # outside the band = passes
    c.save("fig6_two_checks_v3")
    fig6_canvas(d27, iso, full=False)[0].save("fig05j_two_checks", report=True)
    out = pd.concat([tA, tB], ignore_index=True).rename(columns={"adt_marker_short": "antibody"})
    out.round(4).to_csv(HERE / "fig6_two_checks_v3.csv", index=False)
    return out


# ----------------------------------------------------------------------------------------------
# F7: Figure 19 relabelled
# ----------------------------------------------------------------------------------------------

def fig7_data(d27: dict, cohort: dict) -> pd.DataFrame:
    ct = pd.read_csv(rs.diag_dir(D18_SLUG) / "factor_cards_celltype.csv")  # Figure 19's source (D18)
    ct = ct[ct["card_id"] == PC8_CARD]
    wide = ct.pivot_table(index="cell_type", columns="label", values=["mean_oriented_z", "n_cells"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.dropna(subset=["mean_oriented_z_malignant", "mean_oriented_z_normal"]).reset_index()
    cells = cohort["cells"]
    # per-cell recompute = D18 (and D27) pooled means; counts are all 13 patients
    m = cells.groupby(["cell_type", "label"])["PC8"].agg(["size", "mean"]).unstack("label")
    for _, r in wide.iterrows():
        for lab in LABELS:
            assert int(m.loc[r["cell_type"], ("size", lab)]) == int(r[f"n_cells_{lab}"])
            assert abs(m.loc[r["cell_type"], ("mean", lab)] - r[f"mean_oriented_z_{lab}"]) < 1e-4
    d27m = d27["means"].pivot_table(index="cell_type", columns="label", values="mean_oriented_z")
    for _, r in wide.iterrows():
        assert abs(d27m.loc[r["cell_type"], "malignant"] - r["mean_oriented_z_malignant"]) < 1e-4
    mal = cells[cells["label"] == "malignant"]
    comp = pd.crosstab(mal["cell_type"], mal["patient"])
    healthy = cells[cells["label"] == "normal"]
    hcomp = pd.crosstab(healthy["cell_type"], healthy["patient"])
    assert (hcomp.loc[wide["cell_type"]] > 0).sum(axis=1).eq(13).all()  # "its healthy cells come from all 13"
    wide["n_malignant"] = wide["n_cells_malignant"].astype(int)
    wide["n_healthy"] = wide["n_cells_normal"].astype(int)
    wide["top_patient"] = wide["cell_type"].map(comp.idxmax(axis=1))
    wide["top_patient_n"] = wide["cell_type"].map(comp.max(axis=1)).astype(int)
    wide["top_patient_share"] = wide["top_patient_n"] / wide["n_malignant"]
    wide["n_patients_with_malignant"] = wide["cell_type"].map((comp > 0).sum(axis=1)).astype(int)
    wide["malignant_by_patient"] = wide["cell_type"].map(
        lambda t: "; ".join(f"{p} {int(n)}" for p, n in comp.loc[t].sort_values(ascending=False).items() if n > 0))
    wide["malignant_higher"] = wide["mean_oriented_z_malignant"] > wide["mean_oriented_z_normal"]
    wide = wide.sort_values("n_malignant", ascending=False).reset_index(drop=True)
    return wide


def figure7(tab: pd.DataFrame, cohort: dict) -> dict:
    cells = cohort["cells"]
    n_mal_all = int((cells["label"] == "malignant").sum())
    vc = cells[cells["label"] == "malignant"]["cell_type"].value_counts()
    hidden = vc[~vc.index.isin(tab["cell_type"])]
    assert (hidden < CT_MIN_POOLED).all()
    assert tab["n_malignant"].sum() + hidden.sum() == n_mal_all
    n_hi = int(tab["malignant_higher"].sum())
    hsc2 = tab.set_index("cell_type").loc["HSC/MPP2"]
    same = same_type_pairs()
    for full in (True, False):
        fig7_canvas(tab, cohort, hidden, hsc2, n_hi, full, same).save(
            "fig7_celltype_means_v3" if full else "fig05g_celltype_counts", report=not full)
    fig7_core_canvas(tab, n_hi, hsc2).save("fig05m_celltype_core", report=True)
    out = tab[["cell_type", "n_malignant", "n_healthy", "mean_oriented_z_malignant", "mean_oriented_z_normal",
               "malignant_higher", "top_patient", "top_patient_n", "top_patient_share", "n_patients_with_malignant",
               "malignant_by_patient"]].rename(columns={"mean_oriented_z_normal": "mean_oriented_z_healthy"})
    hid = pd.DataFrame({"cell_type": hidden.index, "n_malignant": hidden.values, "n_healthy": np.nan,
                        "malignant_by_patient": "not drawn (< 20 malignant cells)"})
    pd.concat([out, hid], ignore_index=True).round(4).to_csv(HERE / "fig7_celltype_means_v3.csv", index=False)
    return {"n_hi": n_hi, "hidden": hidden, "hsc2": hsc2, "same": same}


def same_type_pairs() -> dict:
    """v2 fig3: malignant minus healthy mean PC8 inside each supported patient x predicted type with >= 10 cells of
    each label (the "19 of 21" that the Figure 19 texts quote)."""
    t = pd.read_csv(HERE / "fig3_malignant_vs_healthy_same_type_v2.csv")
    assert (t["n_malignant"] >= 10).all() and (t["n_healthy"] >= 10).all() and set(t["patient"]) <= set(SUPPORTED)
    pos = t["malignant_minus_healthy"] > 0
    neg = t.loc[~pos, ["patient", "cell_type"]].apply(lambda r: f"{r['patient']} {r['cell_type']}", axis=1)
    assert (int(pos.sum()), len(t)) == (19, 21), (int(pos.sum()), len(t))
    return {"n_pos": int(pos.sum()), "n": len(t), "exceptions": sorted(neg)}


def fig7_canvas(tab: pd.DataFrame, cohort: dict, hidden: pd.Series, hsc2: pd.Series, n_hi: int,
                full: bool, same: dict) -> Canvas:
    """full = the update figure (title, definition, notes, footer); otherwise the report version (plot only)."""
    cells = cohort["cells"]
    n_mal_all = int((cells["label"] == "malignant").sum())
    n_h_all = int((cells["label"] == "normal").sum())
    W = 7.0
    row_h = 0.27
    plot_h = row_h * len(tab) + 0.1
    title = (f"Malignant cells score higher on PC8 than healthy cells of the\nsame predicted type in "
             f"{n_hi} of {len(tab)} types, all 13 patients pooled")
    sub = ("Mean oriented PC8 score: each cell's PC8 score, z-scored over all "
           f"{len(cells):,} cells and multiplied\nby −1 so that positive points to the malignant-leaning "
           "end, then averaged over the cells of one\ntype and label. 0 = the cohort average; +1 = one "
           "cohort standard deviation toward the\nmalignant-leaning end.")
    n_note_lines = 8
    H = (Canvas.header_height(title, sub, REPORT_PLACE["F7"]) + 0.55 + plot_h + 0.5 + 0.145 * n_note_lines + 0.55
         if full else (0.55 + plot_h + 0.5))
    c = Canvas(W, H)
    y0 = 0.0  # report version: the LaTeX caption carries the title and the definition
    if full:
        y0 = c.header(title, sub, REPORT_PLACE["F7"])
    top = y0 + 0.55
    axL = 2.35
    ax = c.ax(axL, top, 3.05, plot_h)
    y = np.arange(len(tab), dtype=float)
    for yi, (_, r) in zip(y, tab.iterrows()):
        ax.plot([r["mean_oriented_z_normal"], r["mean_oriented_z_malignant"]], [yi, yi], color=rs.AXIS, lw=rs.LINE_W,
                zorder=1, solid_capstyle="butt")
    ax.plot(tab["mean_oriented_z_normal"], y, ls="none", marker="o", ms=rs.MARKER, mfc=rs.SURFACE, mec=rs.MUTED,
            mew=1.4, zorder=3)
    ax.plot(tab["mean_oriented_z_malignant"], y, ls="none", marker="o", ms=rs.MARKER, mfc=rs.INK_2, mec=rs.SURFACE,
            mew=rs.RING, zorder=4)
    ax.axvline(0, color=rs.INK_2, lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.cell_type}  ({r.n_malignant:,} / {r.n_healthy:,})" for r in tab.itertuples()],
                       fontsize=FS_SMALL)
    for t, r in zip(ax.get_yticklabels(), tab.itertuples()):
        if not r.malignant_higher:
            t.set_color(rs.MUTED)
    ax.set_ylim(len(tab) - 0.45, -0.55)
    clean_axes(ax)
    ax.set_xlim(-0.45, 1.3)
    ax.set_xticks([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ax.set_xticklabels([f"{MINUS}0.25", "0", "+0.25", "+0.5", "+0.75", "+1.0", "+1.25"], fontsize=FS_SMALL - 0.4)
    ax.set_xlabel("Mean oriented PC8 score (cohort SD)", fontsize=FS_SMALL, color=rs.INK_2)
    c.text(axL - 0.08, top - 0.08, "predicted type  (malignant / healthy cells)", fontsize=FS_SMALL - 0.6,
           color=rs.MUTED, ha="right", va="bottom")
    # right column: where the malignant cells of each type come from
    xr = axL + 3.05 + 0.2
    c.text(xr, top - 0.08, "malignant cells: largest\nshare from one patient", fontsize=FS_SMALL - 0.6,
           color=rs.MUTED, va="bottom", linespacing=1.1)
    for yi, r in zip(y, tab.itertuples()):
        yy = top + (yi + 0.55) / (len(tab) + 0.1) * plot_h
        c.text(xr, yy, f"{r.top_patient} {100 * r.top_patient_share:.0f}%  ({r.top_patient_n} of {r.n_malignant})",
               fontsize=FS_SMALL - 0.6, color=rs.INK_2, va="center")
    handles = [Line2D([], [], ls="none", marker="o", ms=rs.MARKER, mfc=rs.SURFACE, mec=rs.MUTED, mew=1.4),
               Line2D([], [], ls="none", marker="o", ms=rs.MARKER, mfc=rs.INK_2, mec=rs.SURFACE)]
    c.fig.legend(handles, ["healthy cells", "malignant cells"], loc="lower left",
                 bbox_to_anchor=(axL / W, 1 - (top - 0.24) / H), ncol=2, frameon=False, fontsize=FS_SMALL - 0.4,
                 handletextpad=0.3, columnspacing=1.2)
    hidden_txt = ", ".join(f"{t} {n}" for t, n in hidden.items())
    note = (f"(n malignant / n healthy) = cells of that predicted type over all 13 patients. The {len(tab)} types "
            f"shown hold\n{tab['n_malignant'].sum():,} of the {n_mal_all:,} malignant cells and "
            f"{tab['n_healthy'].sum():,} of the {n_h_all:,} healthy cells; types with fewer than {CT_MIN_POOLED} "
            f"malignant cells\nare not drawn ({hidden_txt}). Gray labels: malignant not higher.\n"
            f"Pooled means mix patients: {int(hsc2['top_patient_n'])} of the {int(hsc2['n_malignant'])} malignant "
            f"HSC/MPP2 cells ({100 * hsc2['top_patient_share']:.0f}%) come from {hsc2['top_patient']}, while its "
            f"healthy cells come\nfrom all 13 patients, so a pooled gap can partly be a patient difference. The same "
            "comparison inside each supported patient\nand type (at least 10 cells of each label) is "
            f"fig3_malignant_vs_healthy_same_type_v2: malignant higher in {same['n_pos']} of {same['n']} comparisons\n"
            f"(not in {' or '.join(same['exceptions'])}). PC8 was chosen and oriented by a malignant-vs-healthy probe, "
            "so a higher malignant mean\nis expected; only that the gap persists inside one patient and cell type is "
            "new.")
    assert note.count("\n") + 1 == n_note_lines
    if full:
        c.text(0.1, top + plot_h + 0.5, note, fontsize=FS_NOTE - 0.4, color=rs.INK_2, linespacing=1.35)
        c.footer(FOOTER)
    return c


def fig7_core_canvas(tab: pd.DataFrame, n_hi: int, hsc2: pd.Series) -> Canvas:
    """Core report version of Figure 19 (page 7, Figure 6 bottom; placed at its drawn 6.5 in, like fig05a above
    it): one column per predicted type, healthy (open) and malignant (filled) mean oriented PC8 score, and under
    each type its (malignant / healthy) cell counts and the largest single-patient share of its malignant cells."""
    W, H = 6.5, 1.42
    c = Canvas(W, H)
    fs_tick = FS_SMALL - 1.5
    c.text(0.05, 0.04, "Mean oriented PC8 score by predicted type (13 patients pooled)", fontsize=FS - 0.6,
           fontweight="bold", color=rs.INK)
    axL, axT, axW, axH = 1.0, 0.27, 5.45, 0.72
    ax = c.ax(axL, axT, axW, axH)
    x = np.arange(len(tab), dtype=float)
    for xi, (_, r) in zip(x, tab.iterrows()):
        ax.plot([xi, xi], [r["mean_oriented_z_normal"], r["mean_oriented_z_malignant"]], color=rs.AXIS, lw=rs.LINE_W,
                zorder=1, solid_capstyle="butt")
    ms = rs.MARKER - 1.0
    ax.plot(x, tab["mean_oriented_z_normal"], ls="none", marker="o", ms=ms, mfc=rs.SURFACE, mec=rs.MUTED, mew=1.3,
            zorder=3)
    ax.plot(x, tab["mean_oriented_z_malignant"], ls="none", marker="o", ms=ms, mfc=rs.INK_2, mec=rs.SURFACE,
            mew=rs.RING, zorder=4)
    ax.axhline(0, color=rs.INK_2, lw=0.8, zorder=0)
    ax.set_xlim(-0.5, len(tab) - 0.5)
    ax.set_ylim(-0.45, 1.3)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "+0.5", "+1.0"], fontsize=fs_tick)
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.cell_type}\n{r.n_malignant:,} / {r.n_healthy:,}\n{r.top_patient} "
                        f"{100 * r.top_patient_share:.0f}%" for r in tab.itertuples()], fontsize=fs_tick,
                       linespacing=1.15)
    for t, r in zip(ax.get_xticklabels(), tab.itertuples()):
        t.set_color(rs.INK_2 if r.malignant_higher else rs.MUTED)
    # key for the three lines under each type, right-aligned left of the first column, at the tick labels' height
    # (x in data units, y in axes units; -0.058 = the 3 pt tick pad on this 0.72 in tall axis)
    ax.text(-0.62, -0.058, "type\nmalignant / healthy\ntop patient's share",
            transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=fs_tick, color=rs.MUTED,
            linespacing=1.15)
    handles = [Line2D([], [], ls="none", marker="o", ms=ms, mfc=rs.SURFACE, mec=rs.MUTED, mew=1.3),
               Line2D([], [], ls="none", marker="o", ms=ms, mfc=rs.INK_2, mec=rs.SURFACE)]
    c.fig.legend(handles, ["healthy cells", "malignant cells"], loc="upper right", bbox_to_anchor=(1 - 0.03 / W, 1.0),
                 ncol=2, frameon=False, fontsize=fs_tick + 0.4, handletextpad=0.3, columnspacing=1.0,
                 borderaxespad=0.15)
    assert hsc2["top_patient"] == "P03"
    return c


# ----------------------------------------------------------------------------------------------
# T3: Table 3 with plain headers
# ----------------------------------------------------------------------------------------------

def table3(d27: dict, cohort: dict, iso: pd.DataFrame) -> pd.DataFrame:
    adt = d27["adt"]
    rows = []
    for f, lab, prot, source in T3_ROWS:
        r = d27_row(adt, f, lab, prot)
        g = group_rhos(cohort, d27["pct"], f, lab, prot)
        med_g = float(g["rho"].median())
        assert abs(med_g - r["rho_pct_median"]) < 1e-5, (f, lab, prot, med_g, r["rho_pct_median"])
        # D27's n_pct_same_sign counts groups with the sign of the WITHIN-PATIENT median; check it, then also
        # count groups with the sign of the within-type median itself (the plain reading of "groups agreeing").
        assert int((np.sign(g["rho"]) == np.sign(r["rho_within_median"])).sum()) == r["n_pct_same_sign"]
        k_own = int((np.sign(g["rho"]) == np.sign(med_g)).sum())
        other = d27_row(adt, f, "malignant" if lab == "normal" else "normal", prot)
        ir = iso.loc[(f, lab)]
        rows.append({
            "factor": f, "cells": SHOW[lab], "n_cells_pooled": d27["pooled_n"][lab], "protein": prot,
            "row_source": {"v2": "report Table 3 before 2026-10-10", "v1": "report v1 table",
                           "new": "added 2026-10-10"}[source],
            "mark": ROW_MARK[source].strip(),
            "all_cells_pooled": float(r["rho_pooled"]),
            "median_within_patient": float(r["rho_within_median"]),
            "patients_agreeing": int(r["n_patients_same_sign"]), "n_patients": int(r["n_patients_valid"]),
            "median_within_patient_and_cell_type": med_g,
            "groups_agreeing": k_own, "n_groups": len(g),
            "rho_P03": float(r["rho_P03"]),
            "groups_with_within_patient_sign_d27": int(r["n_pct_same_sign"]),
            "isotype_ref_pooled": ir["iso_pooled"], "isotype_ref_within_patient": ir["iso_within_patient"],
            "isotype_ref_within_type": ir["iso_within_type"],
            "isotype_ref_set_by": f"{ir['iso_pooled_set_by']} / {ir['iso_within_patient_set_by']} / "
                                  f"{ir['iso_within_type_set_by']}",
            "meets_fig20_rule": bool(abs(r["rho_within_median"]) >= ROBUST_ABS
                                     and r["n_patients_same_sign"] >= ROBUST_SAME_SIGN),
            "other_label_median_within_patient": float(other["rho_within_median"]),
            "other_label_patients_agreeing": int(other["n_patients_same_sign"]),
            "other_label_median_within_type": float(other["rho_pct_median"]),
        })
    tab = pd.DataFrame(rows)
    # the sentences of the markdown below that name proteins
    pm = tab[(tab["factor"] == "PC8") & (tab["cells"] == "malignant")].set_index("protein")
    assert set(pm.index[pm["meets_fig20_rule"]]) == {"CD48", "CD116", "CD33", "CD146", "CD63", "CD82"}
    keeps = pm["median_within_patient_and_cell_type"].abs() > pm["isotype_ref_within_type"]
    assert set(pm.index[pm["meets_fig20_rule"] & keeps]) == {"CD63", "CD82"}
    outside = pm["median_within_patient"].abs() > pm["isotype_ref_within_patient"]
    assert (outside | ~pm["meets_fig20_rule"]).all()  # "these six all do" (leave the within-patient band)
    differ = tab[tab["groups_agreeing"] != tab["groups_with_within_patient_sign_d27"]]
    assert set(zip(differ["factor"], differ["cells"], differ["protein"])) == {
        ("PC8", "malignant", "CD33"), ("PC8", "malignant", "GPR56"), ("FA7", "healthy", "CD34"),
        ("FA7", "malignant", "GPR56")}, differ
    tab.round(4).to_csv(HERE / "table3_rho_three_levels_v3.csv", index=False)

    # markdown
    n_groups = {lab: int((d27["pct"]["label"] == lab).sum()) for lab in LABELS}
    n_mal_patients = int(cohort["cells"].loc[cohort["cells"]["label"] == "malignant", "patient"].nunique())
    lines = [
        "# Table 3, v3: PC8 and FA7 against surface protein at three levels",
        "",
        "Spearman ρ between a factor score and a surface protein (CLR), computed among cells of one label only "
        "(healthy or malignant). Both factors are oriented: ρ > 0 means cells ranked higher on the factor (toward "
        "its malignant-leaning end) tend to rank higher on the protein, ρ < 0 that they tend to rank lower (the two "
        "rankings run in reverse). Numbers from D27. " + REPORT_PLACE["T3"] + " The report's version 3 table had "
        "only the unmarked rows. † = in the report v1 table (top 3 by "
        "|pooled ρ|). ‡ = added 2026-10-10: CD116, CD33 and CD146, PC8's other strong malignant-cell correlates "
        "(asked for), and CD63, CD82 and GPR56, the other proteins the two-checks figure labels. Numbers behind it: "
        "`table3_rho_three_levels_v3.csv`.",
        "",
        "| Factor, cells | Protein | all cells pooled | median within patient (patients agreeing) | "
        "median within patient and cell type (groups agreeing) |",
        "|---|---|---|---|---|",
    ]
    for (f, lab), blk in tab.groupby(["factor", "cells"], sort=False):
        n = int(blk["n_cells_pooled"].iloc[0])
        first = True
        for r in blk.itertuples():
            head = f"**{f}, {lab}** ({n:,} cells)" if first else ""
            first = False
            mark = f" {r.mark}" if r.mark else ""
            lines.append(f"| {head} | {r.protein}{mark} | {signed(r.all_cells_pooled, 3)} | "
                         f"{signed(r.median_within_patient, 3)} ({r.patients_agreeing}/{r.n_patients}) | "
                         f"{signed(r.median_within_patient_and_cell_type, 3)} ({r.groups_agreeing}/{r.n_groups}) |")
        r0 = blk.iloc[0]
        lines.append(f"| | *isotype reference* | *{r0['isotype_ref_pooled']:.3f}* | *{r0['isotype_ref_within_patient']:.3f}* | "
                     f"*{r0['isotype_ref_within_type']:.3f}* |")
    lines += [
        "",
        "How to read the columns:",
        "",
        "- **all cells pooled**: one ρ over every cell of that label in the cohort "
        f"({d27['pooled_n']['normal']:,} healthy cells from all 13 patients, or {d27['pooled_n']['malignant']:,} "
        f"malignant cells from the {n_mal_patients} patients that have any).",
        "- **median within patient (patients agreeing)**: ρ computed inside each of the 7 patients with at least 10 "
        "malignant cells, then the median of the 7; in brackets, how many of the 7 have the same sign as that median "
        "(the report's 'within (k/7)').",
        "- **median within patient and cell type (groups agreeing)**: ρ computed inside each patient × predicted cell "
        f"type group with at least {PCT_MIN} cells of that label ({n_groups['normal']} healthy groups from all 7 patients; "
        f"{n_groups['malignant']} malignant groups from P02, P03, P04 and P09 only), then the median; in brackets, how "
        "many groups have the same sign as that median (the report's 'in type' showed the median only). D27's own "
        "count (`n_pct_same_sign`) instead uses the sign of the within-patient median; the CSV has both, and they "
        "differ only where the two medians differ in sign (here PC8 × CD33 and PC8 × GPR56 in malignant cells, FA7 × "
        "CD34 in healthy cells and FA7 × GPR56 in malignant cells).",
        "- **isotype reference**: the largest |ρ| that any of the 9 isotype-control antibodies (no target on human "
        "cells) reaches in the same column, for that factor and label. It is a yardstick, not a significance test. "
        "For PC8 in malignant cells one isotype (RTK2071) sets both the within-patient and the within-type value.",
        "- D27's 'robust' rule (the old report Figure 20, now Figure 21) reads the within-patient column only: "
        "|median| ≥ 0.20 and the same sign in at least 5 of 7 patients. CD48, CD116, CD33, CD146, CD63 and CD82 "
        "pass it for PC8 in malignant cells; of these only CD63 and CD82 keep much inside cell types. The rule's 0.20 "
        "bar is lower than the malignant-cell isotype reference (0.227), so passing it does not by itself mean "
        "leaving the isotype band; these six all do.",
        "",
        "Descriptive; Stage 1 is transductive and fit on RNA only; CLR without dsb; predicted cell types are "
        "reference-mapping labels.",
    ]
    (HERE / "table3_rho_three_levels_v3.md").write_text("\n".join(lines) + "\n")
    return tab


# ----------------------------------------------------------------------------------------------

def cell_type_mix(cohort: dict) -> pd.DataFrame:
    cells = cohort["cells"]
    rows = []
    for lab in LABELS:
        for scope, sel in (("all 13 patients", cells["label"] == lab),
                           ("7 supported patients", (cells["label"] == lab) & cells["patient"].isin(SUPPORTED))):
            vc = cells.loc[sel, "cell_type"].value_counts(normalize=True)
            rows.append({"cells": SHOW[lab], "scope": scope, "n": int(sel.sum()),
                         "top_types": "; ".join(f"{t} {100 * v:.1f}%" for t, v in vc.head(5).items())})
    return pd.DataFrame(rows)


def main() -> None:
    rs.apply()
    pd.set_option("display.width", 250)
    d27 = load_d27()
    iso = isotype_refs(d27["adt"])
    for f, name in (("PC8", PC8_CARD), ("FA7", FA7_CARD)):
        for lab in LABELS:
            assert abs(iso.loc[(f, lab), "iso_pooled"] - d27["cs"].loc[name, f"isotype_max_abs_rho_pooled_{lab}"]) < 1e-9
            assert abs(iso.loc[(f, lab), "iso_within_patient"]
                       - d27["cs"].loc[name, f"isotype_max_abs_rho_within_median_{lab}"]) < 1e-9
    print(iso.round(3).to_string())
    cohort = load_cells(d27["orient"])
    print("PC8 input:", cohort["n_genes"], "genes,", cohort["standardization"])

    d5 = fig5_data(d27, cohort)
    figure5(d5, cohort)
    print(f"F5: pooled {d5['rho_pooled']:.4f}; per patient:", d5["per_patient"].round(3).to_dict("records"))
    print(f"F5: within-patient median {d5['med_p']:.4f} ({d5['k_p']}/7); groups:")
    print(d5["groups"].round(3).to_string())
    print(f"F5: within-type median {d5['med_g']:.4f} ({d5['k_g']}/{len(d5['groups'])}); 8-cell rho {d5['rho_ex']:.4f}")
    print(d5["example"].round(3).to_string())
    print(d5["vec"].round(3).to_string())
    for prot, sg in d5["sign"].items():
        print(f"F5 D: {prot} pooled rho {sg['rho']:+.4f}; mean {prot} rank per PC8 tenth:",
              np.round(sg["bin_y"]).astype(int).tolist(), "(middle rank", (d5["n_mal"] + 1) / 2, ")")

    f6 = figure6(d27, iso)
    for lab in ("malignant", "healthy"):
        t = f6[(f6["cells"] == lab) & ~f6["is_isotype"]]
        print(f"F6 {lab}: outside x band {int(t['outside_x_band'].sum())}, outside y band "
              f"{int(t['outside_y_band'].sum())}, both {int((t['outside_x_band'] & t['outside_y_band']).sum())}: "
              f"{t[t['outside_x_band'] & t['outside_y_band']]['antibody'].tolist()}; fig20 rule "
              f"{int(t['meets_fig20_rule'].sum())}; |y| <= 0.10: {int((t['rho_pct_median'].abs() <= 0.1).sum())} of {len(t)}")
        print("   outside y band:", t[t["outside_y_band"]].sort_values("rho_pct_median")[
            ["antibody", "rho_within_median", "rho_pct_median"]].round(3).to_dict("records"))
        print("   labelled:", t[t["labelled"]][["antibody", "rho_within_median", "n_patients_same_sign", "rho_pct_median",
                                                 "n_pct_same_sign"]].round(3).to_dict("records"))

    tab7 = fig7_data(d27, cohort)
    s7 = figure7(tab7, cohort)
    print(tab7.round(3).to_string())
    print("F7 hidden:", s7["hidden"].to_dict())

    t3 = table3(d27, cohort, iso)
    print(t3.round(3).to_string())
    for prot in OTHER_STRONG:
        r = d27_row(d27["adt"], "PC8", "malignant", prot)
        print(f"PC8 malignant {prot}: per patient", {p: round(float(r[f"rho_{p}"]), 3) for p in SUPPORTED})
    print(cell_type_mix(cohort).to_string())
    # Healthy-cell CD22 and CD34 by predicted type (sign paragraph of the caption)
    cells, clr, names = cohort["cells"], cohort["clr"], cohort["names"]
    h = (cells["label"] == "normal").to_numpy()
    by = pd.DataFrame({"cell_type": cells.loc[h, "cell_type"], "CD22": clr[h, names.index("ADT-CD22")],
                       "CD34": clr[h, names.index("ADT-CD34")], "PC8": cells.loc[h, "PC8"],
                       "FA7": cells.loc[h, "FA7"]}).groupby("cell_type").agg(["mean"]).round(2)
    print(by.sort_values(("CD22", "mean")).to_string())
    assert not OVERFLOW, "\n".join(OVERFLOW)


if __name__ == "__main__":
    main()
