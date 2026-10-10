"""v2 of the Pavan protein update: four simple figures, one idea each, replacing the dense v1 panel.

The v1 figure (make_figure.py -> protein_association_pc8.{png,pdf}) mixed pooled and within-patient
estimates in one panel and compressed the malignant-vs-healthy comparison into "19 of 20". The user
asked for separate, plainer figures. All four are about PC8 = IFN-alpha PCA K=40 factor 8 (Stage 1 is
RNA only), oriented with the D18 sign (x -1) so that positive = the malignant-leaning pole.

  Figure 1  fig1_proteins_by_patient_v2   Which surface proteins track PC8, patient by patient: a
            heatmap of the within-patient Spearman rho (PC8 score vs per-cell CLR protein), one
            column per supported patient, healthy cells on top and malignant cells below. No pooled
            estimate. Bottom row of each block: the largest |rho| any isotype control reaches in that
            patient (the per-patient noise reference). A dagger marks proteins named in the questions
            report's PC8 protein list (top 3 by |pooled rho|).
  Figure 2  fig2_cd34_between_within_v2   Why CD34 tracks PC8: (A) cell types differ in both mean CD34
            and mean PC8; (B) inside one patient's cell type the CD34 vs PC8 rho sits near 0;
            (C) the same shrinkage for the other healthy-cell proteins of Figure 1.
  Figure 3  fig3_malignant_vs_healthy_same_type_v2   What the "19 of 20" of v1 means: (A) one
            comparison drawn out cell by cell; (B) malignant minus healthy mean PC8 for every
            (patient, cell type) pair with >= 10 cells of each label (21 pairs: 19 of 21 higher; v1
            counted 19 of 20 because its panel left out pDC2), with a bootstrap 95% CI over cells.
            The CSV also holds a sample/batch sensitivity (healthy cells restricted to the malignant
            cells' samples or batches; only P09 mixes samples).
  Figure 4  fig4_isotype_noise_floor_v2   How the noise floor is set: within-patient median rho of all
            161 protein antibodies vs the 9 isotype controls; floor = the largest |isotype| value.

Inputs (EXP-relative, read only):
  analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/ (D27): d27_ifna_pc8_fa7_adt.csv,
    d27_card_summary.csv, d27_strata.csv, d27_celltype_means_by_patient.csv.gz
  stage1_dr/panel_id=single_geneset__hallmark_interferon_alpha_response/method=pca/k=40/seed=42/scores.npy
  the cohort cache (study_config cohort_cache_relpath): obs patient, CN.label, predicted.annotation,
    marker; obsm['ADT'] raw counts (CLR per cell as in D27). The raw AnnData is never loaded (AGENTS R8).
Per-cell recomputations (Figure 2 per-group rho, Figure 3 bootstrap) are asserted against D27 first.

Outputs (this directory): fig{1..4}_*_v2.{png,pdf} (PNG 200 dpi); fig{1..4}_*_v2.csv (the numbers
behind each figure); captions are in caption_v2.md (hand-written, numbers from the printout below).

Run (lab env; about 10 s):
  python scripts/mrd_stage0_2/research/team_updates/2026-10-09_protein_association_for_pavan/make_figures_v2.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")  # AGENTS R1: BLAS threads capped at 4 in an agent shell

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
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from sc_classification.utils.paths import add_study_module_paths, study_config  # noqa: E402

add_study_module_paths(workstreams=("W5_factor_interpretation",))
import factor_cards as fc  # noqa: E402  (D18: card ids, per-cell CLR)

SLUG = "factor_modality_assoc_v1"
PANEL = "single_geneset__hallmark_interferon_alpha_response"
PC8_CARD = fc.card_id(PANEL, "pca", 40, 8)
PC8_COLUMN = 7  # factor 8 (1-based) = column 7 of scores.npy
SUPPORTED = list(rs.SUPPORTED)
LABELS = ("normal", "malignant")  # D27 label names; "normal" is called "healthy" in every figure
SHOW = {"normal": "Healthy cells", "malignant": "Malignant cells"}
N_TOP = 10  # proteins with the largest |median within-patient rho| per label (Figure 1)
EARLIER_TABLE = {"normal": ["CD34", "CD22", "CD112"], "malignant": ["CD99", "CD244", "GPR56"]}  # top 3 |pooled|
PCT_MIN = 30  # D27 PCT_MIN_CELLS: patient x cell type groups for within-cell-type rho
PAIR_MIN = 10  # D27 CELLTYPE_MIN_CELLS_PATIENT: per-patient cell-type means (Figure 3 pairs)
V1_PANEL_B_MIN_MALIGNANT = 20  # v1 panel B showed types with >= 20 malignant cells pooled ("19 of 20")
EXAMPLE_PAIR = ("P03", "HSC/MPP2")  # Figure 3A: the pair with the most malignant cells
N_BOOT = 2000
SEED = 20261010

FS = 8.4
FS_SMALL = 7.6
FS_NOTE = 7.4
MINUS = "−"
HEALTHY_FACE = rs.SURFACE
HEALTHY_EDGE = rs.MUTED
MALIGNANT_FACE = rs.INK_2
# Blue/red mean the sign of rho (Figure 1), so Figure 2 uses neutral gray and aqua for its two levels.
MIXED = rs.INK_2  # within patient, cell types mixed
WITHIN_TYPE = rs.AQUA  # within patient x cell type
ISOTYPE = rs.ORANGE  # Figure 4 isotype controls
DAGGER = "†"  # Figure 1: protein named in the questions report's PC8 protein list (top 3 by |pooled rho|)
BAND = "#e6e5df"
DIV_CMAP = LinearSegmentedColormap.from_list("div", [rs.DIV_NEG, rs.DIV_MID, rs.DIV_POS])


def signed(v: float, nd: int = 2) -> str:
    s = f"{v:+.{nd}f}"
    if s in (f"+{0:.{nd}f}", f"-{0:.{nd}f}"):
        s = f"{0:.{nd}f}"
    return s.replace("-", MINUS)


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------

def d27_dir() -> Path:
    return rs.diag_dir(SLUG)


def load_d27() -> dict:
    adt = pd.read_csv(d27_dir() / "d27_ifna_pc8_fa7_adt.csv")
    adt = adt[adt["card_id"] == PC8_CARD].copy()
    assert adt.groupby("label").size().eq(170).all()
    assert adt[adt["is_isotype"]].groupby("label").size().eq(9).all()
    assert (adt["n_patients_valid"] == len(SUPPORTED)).all() and adt["oriented"].all()
    cs = pd.read_csv(d27_dir() / "d27_card_summary.csv").set_index("card_id").loc[PC8_CARD]
    orient = int(cs["orientation_sign"])
    assert orient == -1  # D18 orientation (as in the report and v1)
    st = pd.read_csv(d27_dir() / "d27_strata.csv")
    n_patient = {lab: st[(st["label"] == lab) & (st["kind"] == "patient")].set_index("patient")["n_cells"]
                 .astype(int).to_dict() for lab in LABELS}
    pct = st[st["kind"] == "patient_celltype"]
    bp = pd.read_csv(d27_dir() / "d27_celltype_means_by_patient.csv.gz")
    bp = bp[bp["card_id"] == PC8_CARD].copy()
    return {"adt": adt, "card": cs, "orient": orient, "n_patient": n_patient, "pct_strata": pct, "by_patient": bp}


def load_cells(orient: int) -> dict:
    """Per-cell table in cohort-cache row order: patient, label, cell type, oriented PC8 (cohort z)."""
    import anndata as ad

    exp = rs.exp_dir()
    cfg = study_config()
    adata = ad.read_h5ad(exp / cfg["cohort_cache_relpath"], backed="r")
    obs = adata.obs
    assert adata.n_obs == int(cfg["cohort"]["n_cells"])
    cells = pd.DataFrame({
        "patient": obs["patient"].astype(str).to_numpy(),
        "label": np.where(obs["CN.label"].astype(str).eq("cancer"), "malignant", "normal"),
        "cell_type": obs["predicted.annotation"].astype(str).to_numpy(),
        "sample": obs["sample"].astype(str).to_numpy(),
        "batch": obs["batch"].astype(str).to_numpy(),
    })
    sort_gate = obs["marker"].astype(str).value_counts()
    adt = adata.obsm["ADT"]
    names = [str(c) for c in adt.columns]
    counts = np.asarray(adt.to_numpy(), dtype=np.float64)
    adata.file.close()
    clr = fc.clr_per_cell(counts)
    scores = np.load(exp / "stage1_dr" / f"panel_id={PANEL}" / "method=pca" / "k=40" / "seed=42" / "scores.npy")
    x = np.asarray(scores, dtype=np.float64)[:, PC8_COLUMN]
    cells["pc8"] = orient * (x - x.mean()) / x.std()  # = D27 oriented_cohort_z
    return {"cells": cells, "clr": clr, "names": names, "counts": counts, "sort_gate": sort_gate}


def check_against_d27(d27: dict, cohort: dict) -> None:
    """The per-cell recompute must reproduce D27's per-patient cell-type means and CD34 rhos."""
    cells = cohort["cells"]
    m = (cells.groupby(["label", "patient", "cell_type"]).agg(n=("pc8", "size"), mean=("pc8", "mean"))
         .reset_index())
    j = d27["by_patient"].merge(m, on=["label", "patient", "cell_type"], how="left")
    assert j["n"].notna().all() and (j["n"] == j["n_cells"]).all()
    assert (j["mean"] - j["mean_oriented_z"]).abs().max() < 1e-4
    cd34 = d27["adt"].set_index(["label", "adt_marker_short"]).loc[("normal", "CD34")]
    i34 = cohort["names"].index("ADT-CD34")
    for p in SUPPORTED:
        sel = (cells["label"] == "normal").to_numpy() & (cells["patient"] == p).to_numpy()
        rho = spearmanr(cohort["clr"][sel, i34], cells.loc[sel, "pc8"])[0]
        assert abs(rho - cd34[f"rho_{p}"]) < 1e-5, (p, rho, cd34[f"rho_{p}"])


# ----------------------------------------------------------------------------------------------
# Figure scaffolding (all geometry in inches from the top-left corner)
# ----------------------------------------------------------------------------------------------

OVERFLOW: list = []  # filled by Canvas.save; main() fails if anything sticks out of a figure


class Canvas:
    def __init__(self, w: float, h: float):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))

    def ax(self, left: float, top: float, width: float, height: float):
        return self.fig.add_axes([left / self.w, 1 - (top + height) / self.h, width / self.w, height / self.h])

    def text(self, x: float, y: float, s: str, **kw):
        kw.setdefault("ha", "left")
        kw.setdefault("va", "top")
        kw.setdefault("color", rs.INK_2)
        kw.setdefault("fontsize", FS)
        return self.fig.text(x / self.w, 1 - y / self.h, s, **kw)

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
        return y

    def panel_title(self, x: float, y: float, letter: str, text: str) -> None:
        self.text(x, y, letter, fontsize=10.5, fontweight="bold", color=rs.INK, va="baseline")
        self.text(x + 0.2, y, text, fontsize=FS + 0.6, fontweight="bold", color=rs.INK, va="baseline")

    def footer(self, s: str) -> None:
        self.text(0.1, self.h - 0.08, s, fontsize=FS_NOTE - 0.6, color=rs.MUTED, va="bottom", linespacing=1.3)

    def overflow(self) -> list:
        """Text artists that stick out of the canvas (the page is saved at exactly W x H inches)."""
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

    def save(self, stem: str) -> None:
        bad = self.overflow()
        if bad:
            OVERFLOW.append(f"{stem}: text outside the {self.w} x {self.h} in canvas: {bad}")
        with plt.rc_context({"savefig.bbox": "standard"}):
            self.fig.savefig(HERE / f"{stem}.png", dpi=200)
            self.fig.savefig(HERE / f"{stem}.pdf")
        plt.close(self.fig)


FOOTER = ("Source: D27 (factor_modality_assoc_v1), post-transplant MDS MRD CITE-seq cohort, the 7 patients with "
          "≥10 malignant cells.\nPC8 = IFN-alpha PCA K=40 factor 8, fit on RNA only. Descriptive.")


def clean_axes(ax, grid_x: bool = True) -> None:
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=grid_x)
    ax.tick_params(axis="y", length=0, pad=4)


# ----------------------------------------------------------------------------------------------
# Figure 1: which proteins track PC8, patient by patient
# ----------------------------------------------------------------------------------------------

def fig1_rows(adt: pd.DataFrame, lab: str) -> pd.DataFrame:
    m = adt[(adt["label"] == lab) & ~adt["is_isotype"]].copy()
    m["abs_med"] = m["rho_within_median"].abs()
    m["rank_within"] = m["abs_med"].rank(ascending=False, method="first").astype(int)
    m["rank_pooled"] = m["rho_pooled"].abs().rank(ascending=False, method="first").astype(int)
    earlier = m[m["rank_pooled"] <= 3].sort_values("rank_pooled")["adt_marker_short"].tolist()
    assert earlier == EARLIER_TABLE[lab], (lab, earlier)
    m["named_in_report"] = m["rank_pooled"] <= 3
    rows = m[(m["rank_within"] <= N_TOP) | m["named_in_report"]].copy()
    return rows.sort_values("rho_within_median", ascending=False).reset_index(drop=True)


def fig1_isotype_row(adt: pd.DataFrame, lab: str) -> dict:
    iso = adt[(adt["label"] == lab) & adt["is_isotype"]]
    per_patient = {p: float(iso[f"rho_{p}"].abs().max()) for p in SUPPORTED}
    floor = float(iso["rho_within_median"].abs().max())  # Figure 4's floor (largest |isotype median|)
    row_median = float(np.median(list(per_patient.values())))  # median of this row, like the protein rows
    return {"per_patient": per_patient, "floor": floor, "row_median": row_median}


def draw_heat_block(ax, rows: pd.DataFrame, iso: dict, norm: Normalize) -> None:
    n = len(rows)
    gap = 0.06
    for i, (_, r) in enumerate(rows.iterrows()):
        for j, p in enumerate(SUPPORTED):
            v = float(r[f"rho_{p}"])
            ax.add_patch(Rectangle((j + gap / 2, i + gap / 2), 1 - gap, 1 - gap, facecolor=DIV_CMAP(norm(v)),
                                   edgecolor="none"))
            dark = abs(v) / norm.vmax > 0.62
            ax.text(j + 0.5, i + 0.52, signed(v), ha="center", va="center", fontsize=FS_SMALL - 0.2,
                    color="white" if dark else rs.INK)
        # right-hand summary column
        med = float(r["rho_within_median"])
        ax.text(len(SUPPORTED) + 0.3, i + 0.52, signed(med), ha="left", va="center", fontsize=FS_SMALL,
                color=rs.INK, fontweight="bold")
        ax.text(len(SUPPORTED) + 1.75, i + 0.52, f"{int(r['n_patients_same_sign'])}/{len(SUPPORTED)}", ha="left",
                va="center", fontsize=FS_SMALL, color=rs.INK_2)
    # isotype reference row
    yi = n + 0.35
    for j, p in enumerate(SUPPORTED):
        ax.add_patch(Rectangle((j + gap / 2, yi + gap / 2), 1 - gap, 1 - gap, facecolor="none", edgecolor=rs.AXIS,
                               lw=0.6))
        ax.text(j + 0.5, yi + 0.52, f"{iso['per_patient'][p]:.2f}", ha="center", va="center", fontsize=FS_SMALL - 0.2,
                color=rs.INK_2)
    ax.text(len(SUPPORTED) + 0.3, yi + 0.52, f"{iso['row_median']:.2f}", ha="left", va="center",
            fontsize=FS_SMALL, color=rs.INK_2)
    ax.text(len(SUPPORTED) + 1.75, yi + 0.52, "–", ha="left", va="center", fontsize=FS_SMALL, color=rs.MUTED)
    ax.set_xlim(0, len(SUPPORTED) + 2.6)
    ax.set_ylim(n + 1.4, -0.02)
    ax.set_yticks([i + 0.52 for i in range(n)] + [yi + 0.52])
    labels = [f"{m} {DAGGER}" if e else m for m, e in zip(rows["adt_marker_short"], rows["named_in_report"])]
    ax.set_yticklabels(labels + ["isotype controls:\nlargest |ρ| in that patient"], fontsize=FS)
    for tick in ax.get_yticklabels()[:-1]:
        tick.set_color(rs.INK_2)
    ax.get_yticklabels()[-1].set_color(rs.MUTED)
    ax.get_yticklabels()[-1].set_linespacing(1.0)
    ax.get_yticklabels()[-1].set_fontsize(FS_SMALL)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("none")


def draw_heat_header(c: Canvas, left: float, top: float, cell_w: float, lab: str, n_patient: dict) -> None:
    for j, p in enumerate(SUPPORTED):
        x = left + (j + 0.5) * cell_w
        c.text(x, top, p, ha="center", fontsize=FS, fontweight="bold", color=rs.INK)
        c.text(x, top + 0.17, f"{n_patient[lab][p]:,}", ha="center", fontsize=FS_SMALL - 0.4, color=rs.INK_2)
    x_med = left + (len(SUPPORTED) + 0.3) * cell_w
    c.text(x_med, top, "median", fontsize=FS_SMALL, color=rs.INK_2, fontweight="bold")
    c.text(x_med + 1.45 * cell_w, top, "same\nsign", fontsize=FS_SMALL, color=rs.INK_2, linespacing=1.0)


def figure1(d27: dict) -> pd.DataFrame:
    adt = d27["adt"]
    rows = {lab: fig1_rows(adt, lab) for lab in LABELS}
    iso = {lab: fig1_isotype_row(adt, lab) for lab in LABELS}
    vmax = 0.5
    norm = Normalize(-vmax, vmax)

    W = 7.0
    cell_w, cell_h = 0.54, 0.235
    label_w = 1.55
    block_w = cell_w * (len(SUPPORTED) + 2.6)
    heights = {lab: cell_h * (len(rows[lab]) + 1.4) for lab in LABELS}
    top0 = 1.5  # below the title block (asserted after drawing it)
    head_h = 0.84
    gap_between = 0.3
    H = top0 + 2 * head_h + heights["normal"] + gap_between + heights["malignant"] + 1.15
    c = Canvas(W, H)
    y_head = c.header("Healthy cells: the same few proteins track PC8 in nearly every patient.\n"
                      "Malignant cells: the correlations vary more between patients.",
                      "One square = Spearman ρ between PC8 score and protein level among one patient's\n"
                      "healthy or malignant cells (the number under each patient = cells used).",
                      "PC8 is built from RNA only. ρ > 0 (red): cells with more of the protein sit toward the "
                      "malignant-leaning\npole of PC8. Rows: the 10 proteins with the largest |median|, plus "
                      f"{DAGGER} = named in our report (largest pooled ρ).")
    assert y_head + 0.15 <= top0, y_head
    shrink = rows["normal"]["rho_pct_median"].abs().max()
    cd34_pct = float(rows["normal"].set_index("adt_marker_short").loc["CD34", "rho_pct_median"])
    n_mal = d27["n_patient"]["malignant"]
    block_note = {
        "normal": (f"Mostly a cell-type effect: inside one patient's cell type these shrink to about ±{shrink:.2f} "
                   f"or less (CD34 {signed(cd34_pct)}; Fig. 2)."),
        "malignant": (f"Fewer cells ({min(n_mal.values())} to {max(n_mal.values())} per patient): read each value "
                      "against that patient's isotype row, not against 0."),
    }
    y = top0
    left = label_w + 0.1
    for lab in LABELS:
        c.text(0.1, y + 0.02, SHOW[lab], fontsize=FS + 1.4, fontweight="bold", color=rs.INK)
        c.text(0.1, y + 0.25, block_note[lab], fontsize=FS_NOTE, color=rs.INK_2)
        draw_heat_header(c, left, y + 0.46, cell_w, lab, d27["n_patient"])
        ax = c.ax(left, y + head_h, block_w, heights[lab])
        draw_heat_block(ax, rows[lab], iso[lab], norm)
        y += head_h + heights[lab] + gap_between
    # colour scale
    cb_left, cb_top, cb_w = left, H - 0.98, cell_w * len(SUPPORTED)
    cax = c.ax(cb_left, cb_top, cb_w, 0.11)
    grad = np.linspace(-vmax, vmax, 256)[None, :]
    cax.imshow(grad, aspect="auto", cmap=DIV_CMAP, norm=norm, extent=(-vmax, vmax, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    cax.set_xticklabels([f"≤{MINUS}0.5", f"{MINUS}0.25", "0", "+0.25", "≥+0.5"], fontsize=FS_SMALL - 0.4)
    cax.tick_params(axis="x", length=2, pad=2)
    cax.grid(False)
    for s in cax.spines.values():
        s.set_visible(False)
    c.text(cb_left + cb_w + 0.3, cb_top - 0.02, "Spearman ρ within\nthe patient",
           fontsize=FS_SMALL - 0.2, color=rs.INK_2, linespacing=1.25)
    c.footer(FOOTER)
    c.save("fig1_proteins_by_patient_v2")

    out = []
    for lab in LABELS:
        for _, r in rows[lab].iterrows():
            row = {"cells": "healthy" if lab == "normal" else "malignant", "protein": r["adt_marker_short"],
                   "named_in_report": bool(r["named_in_report"])}
            row.update({f"rho_{p}": round(float(r[f"rho_{p}"]), 3) for p in SUPPORTED})
            row.update({"rho_median": round(float(r["rho_within_median"]), 3),
                        "n_patients_same_sign": int(r["n_patients_same_sign"]),
                        "rho_within_celltype_median": round(float(r["rho_pct_median"]), 3), "floor": np.nan})
            out.append(row)
        row = {"cells": "healthy" if lab == "normal" else "malignant",
               "protein": "isotype controls: max |rho| in that patient", "named_in_report": False}
        row.update({f"rho_{p}": round(iso[lab]["per_patient"][p], 3) for p in SUPPORTED})
        row.update({"rho_median": round(iso[lab]["row_median"], 3), "n_patients_same_sign": np.nan,
                    "rho_within_celltype_median": np.nan, "floor": round(iso[lab]["floor"], 3)})
        out.append(row)
    tab = pd.DataFrame(out)
    tab.to_csv(HERE / "fig1_proteins_by_patient_v2.csv", index=False)
    return tab


# ----------------------------------------------------------------------------------------------
# Figure 2: CD34 between vs within cell types
# ----------------------------------------------------------------------------------------------

def fig2_data(d27: dict, cohort: dict, fig1: pd.DataFrame) -> dict:
    cells = cohort["cells"]
    clr, names = cohort["clr"], cohort["names"]
    i34 = names.index("ADT-CD34")
    healthy = (cells["label"] == "normal").to_numpy() & cells["patient"].isin(SUPPORTED).to_numpy()
    h = cells[healthy].copy()
    h["cd34"] = clr[healthy, i34]
    types = (h.groupby("cell_type").agg(n=("pc8", "size"), mean_cd34_clr=("cd34", "mean"),
                                        mean_pc8=("pc8", "mean")).reset_index())
    types = types[types["n"] >= PCT_MIN].sort_values("mean_cd34_clr", ascending=False).reset_index(drop=True)
    slope, intercept = np.polyfit(types["mean_cd34_clr"], types["mean_pc8"], 1)  # unweighted, one point per type
    rho_types = spearmanr(types["mean_cd34_clr"], types["mean_pc8"])[0]

    groups = []
    for (p, ct), g in h.groupby(["patient", "cell_type"]):
        if len(g) >= PCT_MIN:
            groups.append({"patient": p, "cell_type": ct, "n_cells": len(g),
                           "rho_cd34_pc8": spearmanr(g["cd34"], g["pc8"])[0]})
    groups = pd.DataFrame(groups)
    # Same strata as D27 and the same median.
    pct = d27["pct_strata"]
    pct = pct[pct["label"] == "normal"]
    assert len(groups) == len(pct) == 71
    cd34 = d27["adt"].set_index(["label", "adt_marker_short"]).loc[("normal", "CD34")]
    assert abs(groups["rho_cd34_pc8"].median() - cd34["rho_pct_median"]) < 1e-5
    mixed = pd.DataFrame({"patient": SUPPORTED, "rho_cd34_pc8": [float(cd34[f"rho_{p}"]) for p in SUPPORTED]})

    # Panel C: CD34 plus the other healthy-cell proteins of Figure 1 (within patient vs within cell type).
    adt = d27["adt"]
    prots = fig1[(fig1["cells"] == "healthy") & ~fig1["protein"].str.startswith("isotype")]["protein"].tolist()
    a = adt[(adt["label"] == "normal") & adt["adt_marker_short"].isin(prots)].copy()
    a = a.sort_values("rho_within_median", ascending=False).reset_index(drop=True)
    shrink = a[["adt_marker_short", "rho_within_median", "n_patients_same_sign", "rho_pct_median",
                "n_pct_strata", "n_pct_same_sign"]].rename(columns={"adt_marker_short": "protein"})

    raw = cohort["counts"][:, i34]
    raw_q = np.percentile(raw, [5, 50, 95])
    raw_by_type = (pd.Series(cohort["counts"][healthy, i34]).groupby(h["cell_type"].to_numpy()).median())
    types["median_raw_cd34"] = types["cell_type"].map(raw_by_type)
    gate = cohort["sort_gate"]
    raw_quant = pd.DataFrame({
        "population": "all cells (13 patients, both labels)", "n_cells": len(raw),
        "sort_gate_marker": ";".join(f"{k}={v}" for k, v in gate.items()),
        "raw_cd34_p05": raw_q[0], "raw_cd34_p50": raw_q[1], "raw_cd34_p95": raw_q[2]}, index=[0])
    return {"types": types, "slope": slope, "intercept": intercept, "rho_types": rho_types, "groups": groups,
            "mixed": mixed, "cd34": cd34, "shrink": shrink, "raw_q": raw_q, "raw_by_type": raw_by_type,
            "raw_quant": raw_quant, "n_healthy": int(healthy.sum()), "n_cells_all": len(raw)}


LABEL_OFFSETS = {  # (dx, dy) in points for Figure 2A labels, placed by eye to avoid collisions
    "HSC/MPP1": (0, 9), "HSC/MPP2": (0, 10), "NP1": (7, -6), "MEP-MkP": (-7, 0), "EBM": (-7, 0),
    "pDC1": (7, 0), "LMPP": (-8, 0), "CLP": (9, -4), "MEP-EP": (-10, 6), "NP2": (-8, 0),
    "cDC": (-9, 0), "pDC2": (-9, 0), "ncMono": (7, 0),
}
LABEL_HA = {"HSC/MPP1": "center", "HSC/MPP2": "center", "MEP-MkP": "right", "EBM": "right", "LMPP": "right",
            "MEP-EP": "right", "NP2": "right", "cDC": "right", "pDC2": "right"}


def figure2(d: dict) -> None:
    types, groups, mixed, shrink = d["types"], d["groups"], d["mixed"], d["shrink"]
    W, H = 7.0, 9.3
    c = Canvas(W, H)
    y0 = c.header("CD34 tracks PC8 mostly because cell types differ,\nnot because cells within a type differ",
                  "A: one dot = one predicted cell type (healthy cells of the 7 patients).\n"
                  "B: one dot = CD34 vs PC8 ρ in one patient's healthy cells of one type (gray: types mixed).\n"
                  "C: medians of such ρ for the 10 healthy-cell proteins of Figure 1.",
                  "Every cell passed a CD34+ sort gate, yet CD34 protein still varies widely (raw counts over all "
                  f"{d['n_cells_all']:,}\ncells: median {d['raw_q'][1]:.0f}, 5th to 95th percentile {d['raw_q'][0]:.0f} "
                  f"to {d['raw_q'][2]:.0f}). It is brightest on HSC/MPP and dimmer on\ncommitted progenitors, so "
                  "within the gate CD34 level mostly reports differentiation stage.")
    y0 += 0.25
    top_ax = y0 + 0.36
    # ---- Panel A
    c.panel_title(0.1, y0, "A", "Types differ in both CD34 and PC8")
    axA = c.ax(0.78, top_ax, 2.42, 3.0)
    sizes = 12 + 110 * types["n"] / types["n"].max()
    xs = np.linspace(types["mean_cd34_clr"].min() - 0.15, types["mean_cd34_clr"].max() + 0.15, 50)
    axA.plot(xs, d["intercept"] + d["slope"] * xs, color=rs.AXIS, lw=1.2, zorder=1)
    axA.scatter(types["mean_cd34_clr"], types["mean_pc8"], s=sizes, color=rs.INK_2, edgecolor=rs.SURFACE,
                linewidth=0.9, zorder=3)
    for _, r in types.iterrows():
        dx, dy = LABEL_OFFSETS.get(r["cell_type"], (6, 3))
        ha = LABEL_HA.get(r["cell_type"], "left")
        axA.annotate(r["cell_type"], (r["mean_cd34_clr"], r["mean_pc8"]), xytext=(dx, dy), textcoords="offset points",
                     fontsize=FS_SMALL - 0.4, color=rs.INK_2, ha=ha, va="center")
    axA.set_xlabel("Mean CD34 protein (CLR)", fontsize=FS, color=rs.INK_2)
    axA.set_ylabel("Mean PC8 score\n(→ malignant-leaning pole)", fontsize=FS, color=rs.INK_2)
    axA.grid(axis="both", visible=True)
    axA.set_xlim(-0.1, 4.5)
    axA.set_ylim(-0.95, 0.98)
    axA.axhline(0, color=rs.INK_2, lw=0.6, zorder=0)
    axA.text(0.97, 0.04, f"across the {len(types)} types:\nSpearman ρ = {signed(d['rho_types'])}\n"
                         "line = least-squares fit,\none point per type\nlarger dot = more cells",
             transform=axA.transAxes, fontsize=FS_SMALL - 0.4, color=rs.INK_2, va="bottom", ha="right",
             linespacing=1.3)

    # ---- Panel B
    xB = 4.25
    c.panel_title(3.55, y0, "B", "Inside most types, CD34 barely tracks PC8")
    c.text(3.75, y0 + 0.08, "rows: types by mean CD34, brightest at top; bar = median", fontsize=FS_NOTE - 0.4,
           color=rs.MUTED)
    rows_ct = list(types["cell_type"])
    axB = c.ax(xB, top_ax, 2.15, 3.0)
    ys = {"__mixed__": 0.0}
    for i, ct in enumerate(rows_ct):
        ys[ct] = i + 1.45
    rng = np.random.default_rng(SEED)
    axB.axvline(0, color=rs.INK_2, lw=0.7, zorder=1)
    jit = rng.uniform(-0.17, 0.17, len(mixed))
    axB.scatter(mixed["rho_cd34_pc8"], ys["__mixed__"] + jit, s=22, color=MIXED, edgecolor=rs.SURFACE, linewidth=0.7,
                zorder=3)
    med_mixed = float(np.median(mixed["rho_cd34_pc8"]))
    axB.plot([med_mixed, med_mixed], [-0.38, 0.38], color=rs.INK, lw=1.8, zorder=4, solid_capstyle="butt")
    for ct in rows_ct:
        g = groups[groups["cell_type"] == ct]
        jit = rng.uniform(-0.17, 0.17, len(g))
        axB.scatter(g["rho_cd34_pc8"], ys[ct] + jit, s=16, color=WITHIN_TYPE, edgecolor=rs.SURFACE, linewidth=0.6,
                    zorder=3)
        med = float(g["rho_cd34_pc8"].median())
        axB.plot([med, med], [ys[ct] - 0.36, ys[ct] + 0.36], color=rs.INK, lw=1.5, zorder=4, solid_capstyle="butt")
        axB.text(1.03, ys[ct], f"{len(g)}", transform=axB.get_yaxis_transform(), fontsize=FS_SMALL - 0.4,
                 color=rs.INK_2, va="center")
    axB.text(1.03, ys["__mixed__"], f"{len(mixed)}", transform=axB.get_yaxis_transform(), fontsize=FS_SMALL - 0.4,
             color=rs.INK_2, va="center")
    axB.text(1.03, -0.64, "patients", transform=axB.get_yaxis_transform(), fontsize=FS_SMALL - 0.8,
             color=rs.MUTED, va="bottom")
    axB.axhline(0.72, color=rs.AXIS, lw=0.6)
    axB.set_yticks([ys["__mixed__"]] + [ys[ct] for ct in rows_ct])
    axB.set_yticklabels(["all types mixed"] + rows_ct, fontsize=FS_SMALL)
    axB.get_yticklabels()[0].set_color(MIXED)
    axB.get_yticklabels()[0].set_fontweight("bold")
    axB.set_ylim(ys[rows_ct[-1]] + 1.1, -0.6)
    axB.set_xlim(-0.45, 0.68)
    axB.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
    axB.set_xticklabels([f"{MINUS}0.4", f"{MINUS}0.2", "0", "+0.2", "+0.4", "+0.6"], fontsize=FS_SMALL - 0.2)
    axB.set_xlabel("Spearman ρ, CD34 vs PC8\n(healthy cells, within one patient)", fontsize=FS, color=rs.INK_2)
    clean_axes(axB)
    axB.text(med_mixed, -0.45, f"median {signed(med_mixed)}", fontsize=FS_SMALL - 0.4, color=rs.INK,
             ha="center", va="bottom")
    med_all = float(groups["rho_cd34_pc8"].median())
    axB.text(0.99, 0.006, f"median of all {len(groups)} groups: {signed(med_all)}", transform=axB.transAxes,
             fontsize=FS_SMALL - 0.4, color=rs.INK, ha="right", va="bottom")

    # ---- Panel C
    yC = top_ax + 3.0 + 0.8
    c.panel_title(0.1, yC, "C", "The other proteins that track PC8 in healthy cells shrink the same way")
    n_c = len(shrink)
    axC = c.ax(1.0, yC + 0.2, 3.75, 0.215 * n_c)
    yc = np.arange(n_c, dtype=float)
    for yi, (_, r) in zip(yc, shrink.iterrows()):
        axC.annotate("", xy=(r["rho_pct_median"], yi), xytext=(r["rho_within_median"], yi),
                     arrowprops=dict(arrowstyle="-|>", color=rs.AXIS, lw=1.1, shrinkA=4, shrinkB=3,
                                     mutation_scale=7), zorder=1)
    axC.scatter(shrink["rho_within_median"], yc, s=30, color=MIXED, edgecolor=rs.SURFACE, linewidth=0.8, zorder=3)
    axC.scatter(shrink["rho_pct_median"], yc, s=30, color=WITHIN_TYPE, edgecolor=rs.SURFACE, linewidth=0.8, zorder=3)
    axC.axvline(0, color=rs.INK_2, lw=0.7, zorder=0)
    axC.set_yticks(yc)
    axC.set_yticklabels(shrink["protein"], fontsize=FS_SMALL)
    for t in axC.get_yticklabels():
        if t.get_text() == "CD34":
            t.set_fontweight("bold")
            t.set_color(rs.INK)
    axC.set_ylim(n_c - 0.4, -0.6)
    axC.set_xlim(-0.3, 0.3)
    axC.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    axC.set_xticklabels([f"{MINUS}0.3", f"{MINUS}0.2", f"{MINUS}0.1", "0", "+0.1", "+0.2", "+0.3"],
                        fontsize=FS_SMALL - 0.2)
    axC.set_xlabel("Spearman ρ with PC8 (median), healthy cells", fontsize=FS, color=rs.INK_2)
    clean_axes(axC)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor=MIXED,
                      markeredgecolor=rs.SURFACE),
               Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor=WITHIN_TYPE,
                      markeredgecolor=rs.SURFACE)]
    c.fig.legend(handles, ["within patient, types mixed\n(median of 7 patients)",
                           "within patient and type\n(median of 71 groups\nof ≥30 cells)"],
                 loc="upper left", bbox_to_anchor=(4.95 / W, 1 - (yC + 0.3) / H), frameon=False,
                 fontsize=FS_SMALL - 0.2, handletextpad=0.4, labelspacing=1.0)
    c.footer(FOOTER)
    c.save("fig2_cd34_between_within_v2")

    types.round(4).assign(cells="healthy").to_csv(HERE / "fig2a_celltype_means_v2.csv", index=False)
    d["raw_quant"].to_csv(HERE / "fig2_cd34_raw_quantiles_v2.csv", index=False)
    g = pd.concat([mixed.assign(cell_type="all types mixed", n_cells=np.nan), groups], ignore_index=True)
    g[["patient", "cell_type", "n_cells", "rho_cd34_pc8"]].round(4).to_csv(HERE / "fig2b_cd34_rho_by_group_v2.csv",
                                                                            index=False)
    shrink.round(4).to_csv(HERE / "fig2c_within_vs_celltype_v2.csv", index=False)


# ----------------------------------------------------------------------------------------------
# Figure 3: malignant vs healthy, same patient, same cell type
# ----------------------------------------------------------------------------------------------

def bootstrap_diff(mal: np.ndarray, hea: np.ndarray, rng: np.random.Generator) -> tuple:
    bm = rng.choice(mal, size=(N_BOOT, mal.size), replace=True).mean(axis=1)
    bh = rng.choice(hea, size=(N_BOOT, hea.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(bm - bh, [2.5, 97.5])
    return float(lo), float(hi)


CT_ORDER = ["HSC/MPP1", "HSC/MPP2", "LMPP", "MEP-MkP", "MEP-EP", "NP1", "NP2", "cDC", "pDC2"]


def fig3_data(d27: dict, cohort: dict) -> pd.DataFrame:
    cells = cohort["cells"]
    bp = d27["by_patient"]
    bp = bp[bp["supported_patient"]]
    w = bp.pivot_table(index=["patient", "cell_type"], columns="label", values=["mean_oriented_z", "n_cells"]).dropna()
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    assert (w[["n_cells_malignant", "n_cells_normal"]] >= PAIR_MIN).all().all()
    pooled_mal = cells[cells["label"] == "malignant"]["cell_type"].value_counts()
    rng = np.random.default_rng(SEED)
    rows = []
    for _, r in w.iterrows():
        sel = (cells["patient"] == r["patient"]) & (cells["cell_type"] == r["cell_type"])
        is_mal = sel & (cells["label"] == "malignant")
        is_hea = sel & (cells["label"] == "normal")
        mal = cells.loc[is_mal, "pc8"].to_numpy()
        hea = cells.loc[is_hea, "pc8"].to_numpy()
        assert mal.size == r["n_cells_malignant"] and hea.size == r["n_cells_normal"]
        diff = mal.mean() - hea.mean()
        assert abs(diff - (r["mean_oriented_z_malignant"] - r["mean_oriented_z_normal"])) < 1e-4
        lo, hi = bootstrap_diff(mal, hea, rng)
        # Sensitivity: healthy cells restricted to the samples (and batches) the pair's malignant cells come from.
        mal_samples = set(cells.loc[is_mal, "sample"])
        mal_batches = set(cells.loc[is_mal, "batch"])
        hea_ss = cells.loc[is_hea & cells["sample"].isin(mal_samples), "pc8"].to_numpy()
        hea_sb = cells.loc[is_hea & cells["batch"].isin(mal_batches), "pc8"].to_numpy()
        rows.append({"patient": r["patient"], "cell_type": r["cell_type"], "n_malignant": mal.size,
                     "n_healthy": hea.size, "mean_pc8_malignant": mal.mean(), "mean_pc8_healthy": hea.mean(),
                     "malignant_minus_healthy": diff, "ci95_low": lo, "ci95_high": hi,
                     "in_v1_count": bool(pooled_mal.get(r["cell_type"], 0) >= V1_PANEL_B_MIN_MALIGNANT),
                     "healthy_samples": int(cells.loc[is_hea, "sample"].nunique()),
                     "malignant_samples": len(mal_samples),
                     "one_sample_and_batch": bool(cells.loc[sel, "sample"].nunique() == 1
                                                  and cells.loc[sel, "batch"].nunique() == 1),
                     "n_healthy_same_sample": hea_ss.size,
                     "diff_healthy_same_sample": mal.mean() - hea_ss.mean() if hea_ss.size else np.nan,
                     "n_healthy_same_batch": hea_sb.size,
                     "diff_healthy_same_batch": mal.mean() - hea_sb.mean() if hea_sb.size else np.nan})
    out = pd.DataFrame(rows)
    assert set(out["cell_type"]) <= set(CT_ORDER)
    out["_o"] = out["cell_type"].map({c: i for i, c in enumerate(CT_ORDER)})
    out = out.sort_values(["_o", "patient"]).drop(columns="_o").reset_index(drop=True)
    return out


def figure3(pairs: pd.DataFrame, cohort: dict) -> None:
    # Every qualifying pair is counted (v1's "19 of 20" left out pDC2 . P06 only because v1's panel showed
    # types with >= 20 malignant cells pooled; the caption says so for the user).
    v1 = pairs[pairs["in_v1_count"]]
    n_pos_v1 = int((v1["malignant_minus_healthy"] > 0).sum())
    not_higher = pairs[pairs["malignant_minus_healthy"] <= 0]
    n_pos = int((pairs["malignant_minus_healthy"] > 0).sum())
    assert (~pairs["in_v1_count"]).sum() == 1 and len(not_higher) == 2 and n_pos_v1 == len(v1) - 1
    n_excl0 = int(((pairs["ci95_low"] > 0) | (pairs["ci95_high"] < 0)).sum())

    W = 7.0
    row_h = 0.205
    n_rows = len(pairs)
    n_groups = pairs["cell_type"].nunique()
    plot_h = row_h * (n_rows + 0.55 * (n_groups - 1) + 1.0)
    H = 1.5 + 0.35 + 1.45 + 0.55 + plot_h + 1.6
    c = Canvas(W, H)
    y0 = c.header(f"Malignant cells score higher on PC8 than healthy cells of the same type\n"
                  f"in the same patient: {n_pos} of {len(pairs)} comparisons",
                  "Each comparison takes one patient and one predicted cell type and asks: is the mean PC8 score\n"
                  "of the malignant cells higher than that of the healthy cells? (both groups ≥10 cells)")
    y0 += 0.22

    # ---- Panel A: one comparison, cell by cell
    p_ex, ct_ex = EXAMPLE_PAIR
    cells = cohort["cells"]
    sel = (cells["patient"] == p_ex) & (cells["cell_type"] == ct_ex)
    hea = cells.loc[sel & (cells["label"] == "normal"), "pc8"].to_numpy()
    mal = cells.loc[sel & (cells["label"] == "malignant"), "pc8"].to_numpy()
    c.panel_title(0.1, y0, "A", f"One comparison: {ct_ex} cells of patient {p_ex}")
    axA = c.ax(1.25, y0 + 0.2, 4.15, 1.2)
    rng = np.random.default_rng(SEED)
    for yv, vals, face, edge in ((0, hea, HEALTHY_FACE, HEALTHY_EDGE), (1, mal, MALIGNANT_FACE, rs.SURFACE)):
        jit = rng.uniform(-0.28, 0.28, vals.size)
        axA.scatter(vals, yv + jit, s=7, facecolor=face, edgecolor=edge, linewidth=0.5, alpha=0.85, zorder=2)
        m = vals.mean()
        axA.plot([m, m], [yv - 0.4, yv + 0.4], color=rs.INK, lw=2.0, zorder=4, solid_capstyle="butt")
        # healthy label above its row, malignant label below its row, so each sits by its own bar
        axA.text(m, yv - 0.44 if yv == 0 else yv + 0.44, f"mean {signed(m)}", fontsize=FS_SMALL - 0.3,
                 color=rs.INK, ha="center", va="bottom" if yv == 0 else "top", zorder=5,
                 bbox=dict(boxstyle="square,pad=0.1", facecolor=rs.SURFACE, edgecolor="none"))
    d_ex = mal.mean() - hea.mean()
    axA.set_yticks([0, 1])
    axA.set_yticklabels([f"healthy ({hea.size})", f"malignant ({mal.size})"], fontsize=FS_SMALL)
    axA.set_ylim(1.75, -0.7)
    axA.set_xlim(-3.2, 4.2)
    axA.axvline(0, color=rs.AXIS, lw=0.6, zorder=0)
    axA.set_xlabel("PC8 score of each cell (→ malignant-leaning pole)", fontsize=FS, color=rs.INK_2)
    clean_axes(axA)
    c.text(5.6, y0 + 0.36, f"difference of means\n= {signed(mal.mean())} {MINUS} ({signed(hea.mean())})\n"
                           f"= {signed(d_ex)}\n\nThis number is the\n{ct_ex} · {p_ex}\nrow in B.",
           fontsize=FS_SMALL, color=rs.INK_2, linespacing=1.3)

    # ---- Panel B: every comparison
    yB = y0 + 0.2 + 1.2 + 0.75
    c.panel_title(0.1, yB, "B", "Every comparison: malignant mean minus healthy mean")
    axB = c.ax(1.85, yB + 0.38, 3.3, plot_h)
    ypos, y, prev = [], 0.0, None
    for _, r in pairs.iterrows():
        if prev is not None and r["cell_type"] != prev:
            y += 0.55
        ypos.append(y)
        y += 1.0
        prev = r["cell_type"]
    ypos = np.array(ypos)
    axB.axvline(0, color=rs.INK_2, lw=0.8, zorder=1)
    for yi, (_, r) in zip(ypos, pairs.iterrows()):
        neg = r["malignant_minus_healthy"] <= 0
        axB.plot([r["ci95_low"], r["ci95_high"]], [yi, yi], color=rs.AXIS if not neg else rs.MUTED, lw=1.6,
                 solid_capstyle="round", zorder=2)
        is_ex = (r["patient"], r["cell_type"]) == EXAMPLE_PAIR
        axB.plot(r["malignant_minus_healthy"], yi, linestyle="none", marker="o", markersize=5.6,
                 markerfacecolor=rs.SURFACE if neg else rs.INK, markeredgecolor=rs.INK, markeredgewidth=1.1,
                 zorder=3)
        if is_ex:
            axB.annotate("panel A", (r["ci95_high"], yi), xytext=(6, 0), textcoords="offset points",
                         fontsize=FS_SMALL - 0.6, color=rs.INK_2, va="center")
        axB.text(1.03, yi, f"{int(r['n_malignant'])} / {int(r['n_healthy'])}", transform=axB.get_yaxis_transform(),
                 fontsize=FS_SMALL - 0.4, color=rs.INK_2, va="center")
    axB.set_yticks(ypos)
    axB.set_yticklabels([f"{ct} · {p}" for ct, p in zip(pairs["cell_type"], pairs["patient"])], fontsize=FS_SMALL)
    for t, (_, r) in zip(axB.get_yticklabels(), pairs.iterrows()):
        if r["malignant_minus_healthy"] <= 0:
            t.set_fontweight("bold")
            t.set_color(rs.INK)
    axB.set_ylim(ypos[-1] + 0.7, -0.7)
    axB.set_xlim(-0.85, 1.45)
    axB.set_xticks([-0.5, 0, 0.5, 1.0])
    axB.set_xticklabels([f"{MINUS}0.5", "0", "+0.5", "+1.0"], fontsize=FS_SMALL)
    axB.set_xlabel("Malignant minus healthy mean PC8 score\n← healthy higher        malignant higher →",
                   fontsize=FS, color=rs.INK_2)
    clean_axes(axB)
    axB.text(1.03, -0.75, "cells:\nmal. / healthy", transform=axB.get_yaxis_transform(), fontsize=FS_SMALL - 0.8,
             color=rs.MUTED, va="bottom", linespacing=1.0)
    axB.text(-0.02, -0.75, "cell type · patient", transform=axB.get_yaxis_transform(), fontsize=FS_SMALL - 0.8,
             color=rs.MUTED, va="bottom", ha="right")
    # mark the two comparisons where malignant is not higher, in the right margin
    for yi, (_, r) in zip(ypos, pairs.iterrows()):
        if r["malignant_minus_healthy"] <= 0:
            axB.text(1.27, yi, "not higher", transform=axB.get_yaxis_transform(), fontsize=FS_SMALL - 0.4,
                     color=rs.INK, va="center")
    n_cells_all = len(cells)
    c.text(0.1, yB + 0.38 + plot_h + 0.62,
           "Filled dot: malignant higher; open dot: not higher. Line: 95% bootstrap interval (cells resampled "
           f"within each group,\n{N_BOOT:,} draws); {n_excl0} of {len(pairs)} intervals exclude 0, and rows from "
           "one patient are not independent.\nScores are z-scored over all "
           f"{n_cells_all:,} cells, so +0.5 = half a cohort standard deviation.\n"
           "PC8's sign was set by the malignant-vs-healthy classifier, so malignant cells lean positive overall; "
           "what this\nfigure adds is that the gap persists after matching patient and cell type.",
           fontsize=FS_NOTE, color=rs.INK_2,
           linespacing=1.3)
    c.footer(FOOTER)
    c.save("fig3_malignant_vs_healthy_same_type_v2")
    pairs.round(4).to_csv(HERE / "fig3_malignant_vs_healthy_same_type_v2.csv", index=False)
    lbl = "; ".join(f"{r.cell_type} in {r.patient} {r.malignant_minus_healthy:+.3f}" for r in not_higher.itertuples())
    print(f"Fig 3: {n_pos} of {len(pairs)} pairs malignant higher (v1 types: {n_pos_v1} of {len(v1)}); "
          f"not higher: {lbl}; {n_excl0} of {len(pairs)} CIs exclude 0")
    ss = pairs[~pairs["one_sample_and_batch"]]
    print(f"Fig 3 sensitivity: {len(ss)} pairs mix samples or batches; with healthy cells restricted to the "
          f"malignant cells' samples, malignant higher in {int((pairs['diff_healthy_same_sample'] > 0).sum())} of "
          f"{len(pairs)}; same batch: {int((pairs['diff_healthy_same_batch'] > 0).sum())} of {len(pairs)}")
    print(ss[["patient", "cell_type", "healthy_samples", "malignant_samples", "n_healthy", "n_healthy_same_sample",
              "malignant_minus_healthy", "diff_healthy_same_sample", "n_healthy_same_batch",
              "diff_healthy_same_batch"]].round(3).to_string())


# ----------------------------------------------------------------------------------------------
# Figure 4: isotype noise floor
# ----------------------------------------------------------------------------------------------

def figure4(d27: dict) -> pd.DataFrame:
    adt = d27["adt"]
    W, H = 7.0, 5.5
    c = Canvas(W, H)
    y0 = c.header("The noise floor is the strongest PC8 correlation of any isotype control",
                  "One dot = one antibody: the median over the 7 patients of its within-patient Spearman ρ with PC8\n"
                  "(the 'median' column of Figure 1).",
                  "Isotype controls: 9 antibodies in the panel with no target on human cells (named "
                  "'Isotype-<clone>' in the data);\nthey measure non-specific background.")
    y0 += 0.2
    rng = np.random.default_rng(SEED)
    rows = []
    lane_h = 1.18
    for k, lab in enumerate(LABELS):
        a = adt[adt["label"] == lab]
        prot = a[~a["is_isotype"]]
        iso = a[a["is_isotype"]].copy()
        iso["abs"] = iso["rho_within_median"].abs()
        floor = float(iso["abs"].max())
        setter = iso.loc[iso["abs"].idxmax(), "adt_marker_short"].replace("Isotype-", "")
        second = float(iso["abs"].nlargest(2).iloc[-1])
        n_out = int((prot["rho_within_median"].abs() > floor).sum())
        top = y0 + k * (lane_h + 0.55)
        c.text(0.1, top, f"{SHOW[lab]}", fontsize=FS + 1.2, fontweight="bold", color=rs.INK)
        c.text(1.45, top + 0.01, f"floor ±{floor:.2f} (set by {setter}); "
                                 f"{n_out} of {len(prot)} proteins lie outside it",
               fontsize=FS_SMALL, color=rs.INK_2)
        ax = c.ax(1.45, top + 0.25, 4.6, lane_h - 0.25)
        ax.axvspan(-floor, floor, color=BAND, lw=0, zorder=0)
        ax.axvline(0, color=rs.INK_2, lw=0.7, zorder=1)
        prot = prot.assign(y=rng.uniform(-0.22, 0.22, len(prot)))
        ax.scatter(prot["rho_within_median"], prot["y"], s=13, color=rs.MUTED, edgecolor=rs.SURFACE, linewidth=0.5,
                   alpha=0.9, zorder=2)
        iso["y"] = 1 + rng.uniform(-0.12, 0.12, len(iso))
        ax.scatter(iso["rho_within_median"], iso["y"], s=26, color=ISOTYPE, edgecolor=rs.SURFACE, linewidth=0.8,
                   zorder=3)
        if (iso["rho_within_median"] > 0).all():
            ax.text(-0.012, 1, f"all {len(iso)} sit\nright of 0 →", fontsize=FS_SMALL - 0.6, color=rs.INK_2,
                    ha="right", va="center", linespacing=1.05)
        r_set = iso.loc[iso["abs"].idxmax()]
        ax.annotate(setter, (r_set["rho_within_median"], r_set["y"]), xytext=(6, 0), textcoords="offset points",
                    fontsize=FS_SMALL - 0.4, color=rs.INK, ha="left", va="center")
        # name the strongest protein on each side, next to its dot
        for side in (1, -1):
            sd = prot[np.sign(prot["rho_within_median"]) == side]
            r = sd.loc[sd["rho_within_median"].abs().idxmax()]
            ax.annotate(r["adt_marker_short"], (r["rho_within_median"], r["y"]), xytext=(5 * side, 0),
                        textcoords="offset points", fontsize=FS_SMALL - 0.4, color=rs.INK_2,
                        ha="left" if side > 0 else "right", va="center")
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"{len(prot)} protein antibodies", f"{len(iso)} isotype controls"], fontsize=FS_SMALL)
        ax.get_yticklabels()[1].set_color(ISOTYPE)
        ax.set_ylim(1.55, -0.4)
        ax.set_xlim(-0.4, 0.4)
        ax.set_xticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4])
        ax.set_xticklabels([signed(t, 1) if t != 0 else "0" for t in ax.get_xticks()], fontsize=FS_SMALL - 0.2)
        clean_axes(ax)
        if lab == "malignant":
            ax.set_xlabel("Median within-patient Spearman ρ with PC8", fontsize=FS, color=rs.INK_2)
            # the floor if the one isotype that sets it were dropped
            for xv in (-second, second):
                ax.axvline(xv, color=rs.INK_2, lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.text(-second - 0.008, 1.0, f"dashed: floor\nwithout {setter}\n(±{second:.2f})",
                    fontsize=FS_SMALL - 0.6, color=rs.INK_2, ha="right", va="center", linespacing=1.05)
        for _, r in iso.iterrows():
            rows.append({"cells": "healthy" if lab == "normal" else "malignant", "antibody": r["adt_marker_short"],
                         "is_isotype": True, "rho_within_median": round(float(r["rho_within_median"]), 4),
                         "floor": round(floor, 4), "sets_floor": r["adt_marker_short"] == r_set["adt_marker_short"]})
        rows.append({"cells": "healthy" if lab == "normal" else "malignant", "antibody": "proteins outside the floor",
                     "is_isotype": False, "rho_within_median": np.nan, "floor": round(floor, 4),
                     "sets_floor": False, "n_proteins_outside": n_out, "n_proteins": len(prot),
                     "floor_without_setter": round(second, 4)})
    c.text(1.45, H - 0.72, "Gray band: inside the floor, a protein's correlation is no larger than one that an antibody "
                           "with no target\nreaches. The floor is a rough yardstick we chose, not a significance test.",
           fontsize=FS_NOTE, color=rs.INK_2, linespacing=1.3)
    c.footer(FOOTER)
    c.save("fig4_isotype_noise_floor_v2")
    tab = pd.DataFrame(rows)
    tab.to_csv(HERE / "fig4_isotype_noise_floor_v2.csv", index=False)
    return tab


# ----------------------------------------------------------------------------------------------

def main() -> None:
    rs.apply()
    d27 = load_d27()
    cohort = load_cells(d27["orient"])
    check_against_d27(d27, cohort)
    pd.set_option("display.width", 250)
    print("sort gate (obs 'marker'):", cohort["sort_gate"].to_dict())

    fig1 = figure1(d27)
    print(fig1.to_string())
    adt = d27["adt"]
    for lab in LABELS:  # the proteins just past Figure 1's cut at 10 (named in the caption)
        m = adt[(adt["label"] == lab) & ~adt["is_isotype"]].copy()
        m = m.reindex(m["rho_within_median"].abs().sort_values(ascending=False).index)
        print(f"Fig 1 {lab}: ranks {N_TOP + 1}-{N_TOP + 4} by |median|:",
              m.iloc[N_TOP:N_TOP + 4][["adt_marker_short", "rho_within_median", "n_patients_same_sign"]]
              .round(3).to_dict("records"))

    d2 = fig2_data(d27, cohort, fig1)
    figure2(d2)
    print("Fig 2 raw CD34 counts, all cells: 5th/50th/95th percentile", d2["raw_q"])
    print("Fig 2 raw CD34 median by type (healthy, supported):", d2["raw_by_type"].round(0).to_dict())
    print(d2["types"].round(3).to_string())
    print(f"Fig 2A: Spearman across types {d2['rho_types']:.3f}; n healthy cells {d2['n_healthy']}")
    print("Fig 2B: per-type median within-group rho:")
    print(d2["groups"].groupby("cell_type")["rho_cd34_pc8"].agg(["median", "min", "max", "count"]).round(3).to_string())
    print(f"Fig 2B: overall median of {len(d2['groups'])} groups {d2['groups']['rho_cd34_pc8'].median():.4f}; "
          f"types-mixed median {d2['mixed']['rho_cd34_pc8'].median():.4f}")
    print(d2["shrink"].round(3).to_string())

    pairs = fig3_data(d27, cohort)
    figure3(pairs, cohort)
    print(pairs.round(3).to_string())

    fig4 = figure4(d27)
    print(fig4.to_string())
    assert not OVERFLOW, "\n".join(OVERFLOW)


if __name__ == "__main__":
    main()
