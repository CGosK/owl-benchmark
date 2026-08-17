"""Publication figures for the OWL paper. Reads results/*.json and writes
results/figures/*.pdf (vector, Times font to match the ECCV template).

  fig1_design.pdf        : factorial design + pipeline schematic
  fig2_probes.pdf        : S/O/A probe AUCs across layers and test cells
  fig3_causal_heatmap.pdf: intervention x behavior causal specificity matrix
  fig4_monitor.pdf       : monitor false-positive comparison (M vs M_raw vs vVAL)
  fig5_cosines.pdf       : candidate welfare direction vs factorial directions

Every figure is generated fully automatically from the results files; no manual
editing is required. If a file is missing, the script exits with a clear error.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RESULTS  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def load(name):
    p = RESULTS / name
    if not p.exists():
        raise FileNotFoundError(f"missing result file: {p} (run the pipeline first)")
    return json.load(open(p))


def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.axis("off")
    y = 1.0
    ax.text(0.02, y, "OWL benchmark: 2 x 2 x 2 factorial over scenario templates",
            fontweight="bold", fontsize=10)
    y -= 0.14
    for label, x in (("SELF outcome (S)", 0.02), ("OTHER outcome (O)", 0.36),
                     ("surface AFFECT (A)", 0.70)):
        b = plt.Rectangle((x, y - 0.045), 0.28, 0.09, facecolor="#EAF2F8",
                          edgecolor="#0072B2", lw=1)
        ax.add_patch(b)
        ax.text(x + 0.14, y, label, ha="center", va="center", fontsize=8)
    y -= 0.22
    ax.text(0.02, y, "prompt -> residual stream h_l at anchor token", fontsize=9)
    y -= 0.14
    for label, x in (("factorial directions\nv_S, v_O, v_A", 0.02),
                     ("linear probes\nS / O / A", 0.36),
                     ("candidate welfare dirs\nv_VAL, v_ALL", 0.70)):
        b = plt.Rectangle((x, y - 0.06), 0.28, 0.13, facecolor="#FBF3E4",
                          edgecolor="#D55E00", lw=1)
        ax.add_patch(b)
        ax.text(x + 0.14, y, label, ha="center", va="center", fontsize=7.5)
    y -= 0.26
    ax.text(0.02, y, "counterfactual interchange:  h + P_B(h(x') - h(x))   (natural pairs)", fontsize=9)
    y -= 0.14
    for label, x in (("self choice", 0.02), ("other choice", 0.27),
                     ("affect judgment", 0.52), ("capability", 0.77)):
        b = plt.Rectangle((x, y - 0.045), 0.21, 0.09, facecolor="#E8F5E9",
                          edgecolor="#009E73", lw=1)
        ax.add_patch(b)
        ax.text(x + 0.105, y, label, ha="center", va="center", fontsize=7.5)
    y -= 0.2
    ax.text(0.02, y, "ownership-conditioned monitor:  M(x) = P(V- | h) x P(S | V-, h)", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(-0.1, y + 0.02)
    fig.savefig(FIG / "fig1_design.pdf")
    plt.close(fig)


def fig2():
    pres = load("predictive/probe_results.json")
    base = load("predictive/baseline_results.json")
    layers = [int(l) for l in pres["cells"].keys()]
    chosen = int(pres["chosen_layer"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    # left: layer x target probe AUC on test_new
    ax = axes[0]
    x = np.arange(len(layers))
    w = 0.27
    for j, tgt in enumerate(("S", "O", "A")):
        vals = [pres["cells"][str(l)]["test_new"][tgt] for l in layers]
        bars = ax.bar(x + (j - 1) * w, vals, w, color=CB[j], label=tgt)
        for l_i, l in enumerate(layers):
            if l == chosen:
                bars[l_i].set_edgecolor("black")
                bars[l_i].set_linewidth(1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}\n({25*(i+1)-25*0 if False else ''})" for i, l in enumerate(layers)])
    ax.set_xticklabels([f"{l + 1}/{36}" for l in layers])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC (test_new)")
    ax.set_title("(a) Probe AUROC by layer")
    ax.legend(frameon=False, ncol=3)
    # right: chosen-layer probe vs surface baselines on test_new
    ax = axes[1]
    cell = pres["cells"][str(chosen)]["test_new"]
    bc = base["cells"]["test_new"]
    labels = ["probe S", "probe O", "probe A", "TF-IDF S", "TF-IDF O", "TF-IDF A",
              "pronoun S", "vVAL S", "vVAL O", "vVAL A"]
    vals = [cell["S"], cell["O"], cell["A"], bc["tfidf_S"], bc["tfidf_O"],
            bc["tfidf_A"], bc["pronoun_S"], bc["vVAL_raw_S"], bc["vVAL_raw_O"],
            bc["vVAL_raw_A"]]
    colors = [CB[0], CB[1], CB[2], "#888888", "#888888", "#888888",
              "#AAAAAA", CB[3], CB[3], CB[3]]
    ypos = np.arange(len(labels))[::-1]
    ax.barh(ypos, vals, color=colors, height=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUROC (test_new)")
    ax.set_title(f"(b) Chosen layer L{chosen + 1}: probes vs surface baselines")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_probes.pdf")
    plt.close(fig)


def fig3():
    cres = load("causal/causal_results.json")
    summary = cres["summary"]
    intervs = ["S", "O", "A", "FULL", "RANDOM_1", "RANDOM_2"]
    tasks = ["SELF_CHOICE", "OTHER_CHOICE", "AFFECT_JUDGE", "CAPABILITY"]
    M = np.zeros((len(intervs), len(tasks)))
    for i, iv in enumerate(intervs):
        for j, tk in enumerate(tasks):
            key = f"{iv}|{tk}"
            M[i, j] = summary.get(key, {}).get("recovery", np.nan)
    M2 = np.nan_to_num(M, nan=0.0)
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    vmax = max(1.0, float(np.nanmax(np.abs(M2)))) if np.isfinite(np.abs(M2)).any() else 1.0
    im = ax.imshow(M2, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(["self\nchoice", "other\nchoice", "affect\njudgment", "capability"], fontsize=7.5)
    ax.set_yticks(range(len(intervs)))
    ax.set_yticklabels(["S sub-\nspace", "O sub-\nspace", "A sub-\nspace",
                        "full\nvector", "random\n1", "random\n2"], fontsize=7.5)
    for i in range(len(intervs)):
        for j in range(len(tasks)):
            v = M[i, j]
            txt = "nan" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.5 * vmax else "black")
    ax.set_title("Causal recovery: patch-induced / natural counterfactual")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_causal_heatmap.pdf")
    plt.close(fig)


def fig4():
    mres = load("monitor/monitor_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    labels = ["M (ownership)", "M_raw (valence only)", "v_VAL (raw dir)"]
    vals = [mres["distress"]["FPR_M"], mres["distress"]["FPR_M_raw"],
            mres["distress"]["FPR_vVAL"]]
    ax.bar(labels, vals, color=[CB[0], CB[1], CB[3]], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("False-positive rate @ 80% SELF TPR")
    ax.set_title("(a) All distress hard-negatives")
    ax.set_ylim(0, 1.0)
    ax = axes[1]
    kinds = mres["distress_by_kind"]
    kx = list(kinds.keys())
    x = np.arange(len(kx))
    w = 0.27
    for j, (nm, key) in enumerate((("M", "FPR_M"), ("M_raw", "FPR_M_raw"), ("vVAL", "FPR_vVAL"))):
        vals = [kinds[k][key] for k in kx]
        ax.bar(x + (j - 1) * w, vals, w, color=[CB[0], CB[1], CB[3]][j], label=nm)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in kx], fontsize=6.5)
    ax.set_ylabel("FPR @ 80% SELF TPR")
    ax.set_title("(b) By hard-negative kind")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_monitor.pdf")
    plt.close(fig)


def fig5():
    pres = load("predictive/probe_results.json")
    layers = [int(l) for l in pres["cells"].keys()]
    keys = ["cos_vALL_vS", "cos_vALL_vO", "cos_vALL_vA", "cos_vVAL_vS",
            "cos_vVAL_vO", "cos_vVAL_vA"]
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    x = np.arange(len(layers))
    w = 0.13
    for j, k in enumerate(keys):
        vals = [pres["directions"][str(l)][k] for l in layers]
        ax.bar(x + (j - 2.5) * w, vals, w, color=CB[j % 6],
               label=k.replace("cos_", "cos(") + ")")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l + 1}" for l in layers])
    ax.set_xlabel("layer index (1-based)")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Candidate welfare directions vs factorial directions")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_cosines.pdf")
    plt.close(fig)


def main():
    fig1(); print("wrote fig1_design.pdf")
    fig2(); print("wrote fig2_probes.pdf")
    fig3(); print("wrote fig3_causal_heatmap.pdf")
    fig4(); print("wrote fig4_monitor.pdf")
    fig5(); print("wrote fig5_cosines.pdf")


if __name__ == "__main__":
    main()
