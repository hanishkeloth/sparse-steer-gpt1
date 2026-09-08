"""Build figures + tables from results/summary.json and vectors/*.safetensors."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

# categorical palette (fixed order) + text tokens
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 2,
    "lines.markersize": 5,
})


def blues(n):
    """Sequential ramp: one hue, light -> dark."""
    return [matplotlib.colors.to_hex(c) for c in plt.cm.Blues(np.linspace(0.3, 1.0, n))]


def main():
    S = json.load(open(os.path.join(RES, "summary.json")))
    meta = json.load(open(os.path.join(ROOT, "vectors", "meta.json")))
    vecs = load_file(os.path.join(ROOT, "vectors", "gpt1_sentiment_vectors.safetensors"))
    base_nll = S["baseline"]["nll_mean"]
    base_vader = S["baseline"]["vader_mean"]
    out = {"baseline": S["baseline"], "selected": S["stage_a_selected"]}

    # ---------- Figure 1: where does sentiment live? (contrast-vector norm per layer)
    layers = sorted(int(k) for k in meta["layers"])
    dn = [meta["layers"][str(L)]["diff_norm"] for L in layers]
    rn = [meta["layers"][str(L)]["resid_norm"] for L in layers]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.bar(layers, np.array(dn) / np.array(rn), color=BLUE, width=0.7)
    ax.set_xlabel("block (output of)")
    ax.set_ylabel("||mean diff|| / mean ||h||")
    ax.set_title("Sentiment contrast at the shared '.' token, per block", loc="left", color=INK)
    ax.set_xticks(layers)
    fig.savefig(os.path.join(FIG, "fig1_contrast_by_layer.png"))
    plt.close(fig)
    out["contrast_by_layer"] = {str(L): {"diff_norm": d, "resid_norm": r, "ratio": d / r} for L, d, r in zip(layers, dn, rn)}

    # ---------- Figure 2: stage A, effect and fluency cost vs strength per layer
    A = S["stage_a"]
    la = sorted(set(r["layer"] for r in A))
    cs = sorted(set(r["c"] for r in A))
    cols = dict(zip(la, blues(len(la))))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for L in la:
        rows = sorted([r for r in A if r["layer"] == L], key=lambda r: r["c"])
        axes[0].plot([r["c"] for r in rows], [r["effect"] for r in rows], "-o", color=cols[L], label=f"block {L}")
        axes[1].plot([r["c"] for r in rows], [r["nll_cost"] - base_nll for r in rows], "-o", color=cols[L], label=f"block {L}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(cs)
        ax.set_xticklabels([str(c) for c in cs])
        ax.set_xlabel("strength c   (β = c · mean ||h||)")
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[0].set_ylabel("steering effect  (VADER₊ − VADER₋)")
    axes[0].set_title("Effect of the full vector", loc="left")
    axes[1].set_ylabel("Δ NLL vs. unsteered (nats/token)")
    axes[1].set_title("Fluency cost", loc="left")
    axes[1].axhline(0.6, color=INK2, lw=0.8, ls="--")
    axes[1].text(cs[0], 0.62, "selection budget", color=INK2, fontsize=7)
    axes[0].legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(FIG, "fig2_layer_strength.png"))
    plt.close(fig)
    out["stage_a"] = [{k: r[k] for k in ("layer", "c", "beta", "effect", "nll_cost")} for r in A]

    # ---------- Figure 3: sparsity sweep
    B = S["stage_b"]
    if B:
        lb = sorted(set(r["layer"] for r in B))
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        style = {"topk": (BLUE, "top-k of v (magnitude kept)"), "signk": (ORANGE, "sign-k (1-bit, ±1 on top-k)"),
                 "randk": (INK2, "random-k of v (control)")}
        table = {}
        for L in lb:
            ls = "-" if L == lb[0] else "--"
            for kind in ["topk", "signk", "randk"]:
                rows = [r for r in B if r["layer"] == L and r["kind"] == kind]
                if not rows:
                    continue
                ks = sorted(set(r["k"] for r in rows))
                eff = [np.mean([r["effect"] for r in rows if r["k"] == k]) for k in ks]
                eff_lo = [np.min([r["effect"] for r in rows if r["k"] == k]) for k in ks]
                eff_hi = [np.max([r["effect"] for r in rows if r["k"] == k]) for k in ks]
                cost = [np.mean([r["nll_cost"] for r in rows if r["k"] == k]) - base_nll for k in ks]
                c, lab = style[kind]
                axes[0].plot(ks, eff, ls, marker="o", color=c, label=f"{lab}, block {L}")
                if kind == "randk":
                    axes[0].fill_between(ks, eff_lo, eff_hi, color=c, alpha=0.12, lw=0)
                axes[1].plot(ks, cost, ls, marker="o", color=c, label=f"{lab}, block {L}")
                for k, e, co in zip(ks, eff, cost):
                    table.setdefault(str(L), {}).setdefault(kind, {})[str(k)] = {"effect": float(e), "nll_cost": float(co)}
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("k = number of non-zero dimensions (of 768)")
            ax.invert_xaxis()
        axes[0].axhline(0, color=INK2, lw=0.8)
        axes[0].set_ylabel("steering effect  (VADER₊ − VADER₋)")
        axes[0].set_title("Effect vs. sparsity (norm-matched)", loc="left")
        axes[1].set_ylabel("Δ NLL vs. unsteered (nats/token)")
        axes[1].set_title("Fluency cost vs. sparsity", loc="left")
        h, l = axes[0].get_legend_handles_labels()
        fig.subplots_adjust(bottom=0.32, wspace=0.3)
        fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
        fig.savefig(os.path.join(FIG, "fig3_sparsity.png"))
        plt.close(fig)
        out["stage_b"] = table

    # ---------- Figure 4: energy concentration + which dims
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    late = [L for L in layers if L >= 6]
    cols = dict(zip(late, blues(len(late))))
    for L in late:
        v = vecs[f"layer_{L}"].abs().sort(descending=True).values
        frac = (v ** 2).cumsum(0) / (v ** 2).sum()
        axes[0].plot(np.arange(1, 769), frac.numpy(), color=cols[L], label=f"block {L}")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("top-k dimensions")
    axes[0].set_ylabel("fraction of ||v||² captured")
    axes[0].set_title("How concentrated is the steering vector?", loc="left")
    axes[0].legend(fontsize=7)
    # overlap of top-16 sets between blocks
    tops = {L: set(meta["layers"][str(L)]["top16_dims"]) for L in late}
    M = np.array([[len(tops[a] & tops[b]) / 16 for b in late] for a in late])
    im = axes[1].imshow(M, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(late)))
    axes[1].set_xticklabels(late)
    axes[1].set_yticks(range(len(late)))
    axes[1].set_yticklabels(late)
    axes[1].grid(False)
    axes[1].set_title("Overlap of top-16 dims across blocks", loc="left")
    for i in range(len(late)):
        for j in range(len(late)):
            axes[1].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                         color="white" if M[i, j] > 0.5 else INK)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(FIG, "fig4_concentration.png"))
    plt.close(fig)
    out["energy_top_k"] = {}
    for L in late:
        v = vecs[f"layer_{L}"].abs().sort(descending=True).values
        frac = (v ** 2).cumsum(0) / (v ** 2).sum()
        out["energy_top_k"][str(L)] = {str(k): float(frac[k - 1]) for k in [1, 3, 6, 12, 24, 48, 96, 192, 384]}
    # are top dims the high-variance ("outlier") dims?
    out["top_dims_vs_std"] = {}
    for L in late:
        v = vecs[f"layer_{L}"]
        sd = vecs[f"layer_{L}_std"]
        top = torch.topk(v.abs(), 12).indices
        rank_of_top = [int((sd > sd[i]).sum()) + 1 for i in top]  # 1 = highest std dim
        out["top_dims_vs_std"][str(L)] = {"top12_dims": top.tolist(), "std_rank_of_top12": rank_of_top,
                                          "spearman_absv_vs_std": float(np.corrcoef(np.argsort(np.argsort(v.abs().numpy())),
                                                                                    np.argsort(np.argsort(sd.numpy())))[0, 1])}
    json.dump(out, open(os.path.join(RES, "analysis.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("baseline", "selected")}, indent=1))
    if "stage_b" in out:
        for L, kinds in out["stage_b"].items():
            for kind, ks in kinds.items():
                print(L, kind, {k: round(v["effect"], 3) for k, v in ks.items()})


if __name__ == "__main__":
    main()


def cosine_figure():
    """Figure 5: does effect retention equal the cosine between the sparse and the full direction?"""
    import sys as _sys
    S = json.load(open(os.path.join(RES, "summary.json")))
    vecs = load_file(os.path.join(ROOT, "vectors", "gpt1_sentiment_vectors.safetensors"))
    _sys.path.insert(0, os.path.dirname(__file__))
    from common import make_vector
    B = S["stage_b"]
    pts = []
    for L in sorted(set(r["layer"] for r in B)):
        v = vecs[f"layer_{L}"]
        u_full = v / v.norm()
        full = np.mean([r["effect"] for r in B if r["layer"] == L and r["kind"] == "topk" and r["k"] == 768])
        for r in B:
            if r["layer"] != L:
                continue
            rng = np.random.default_rng(1000 * r.get("rand_seed", 0) + r["k"]) if r["kind"] == "randk" else None
            u = make_vector(v, r["kind"], r["k"], rng=rng)
            pts.append({"layer": L, "kind": r["kind"], "k": r["k"], "cos": float(u @ u_full), "retention": r["effect"] / full})
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    style = {"topk": (BLUE, "o", "top-k"), "signk": (ORANGE, "s", "sign-k (1-bit)"), "randk": (INK2, "^", "random-k")}
    for kind, (c, m, lab) in style.items():
        P = [p for p in pts if p["kind"] == kind]
        ax.scatter([p["cos"] for p in P], [p["retention"] for p in P], c=c, marker=m, s=22, label=lab, alpha=0.85, edgecolors="white", linewidths=0.5)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("cosine(sparse direction, full direction)")
    ax.set_ylabel("effect retained (fraction of full vector)")
    ax.set_title("Retention tracks cosine with the dense vector", loc="left")
    ax.legend(fontsize=7, loc="upper left")
    x = np.array([p["cos"] for p in pts]); y = np.array([p["retention"] for p in pts])
    r = float(np.corrcoef(x, y)[0, 1])
    slope = float((x * y).sum() / (x * x).sum())
    ax.text(0.98, 0.05, f"r = {r:.2f}, slope through origin = {slope:.2f}", transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
    fig.savefig(os.path.join(FIG, "fig5_cosine.png"))
    plt.close(fig)
    A = json.load(open(os.path.join(RES, "analysis.json")))
    A["cosine_retention"] = {"points": pts, "pearson_r": r, "slope_through_origin": slope}
    json.dump(A, open(os.path.join(RES, "analysis.json"), "w"), indent=1)
    print("cosine-retention r=", round(r, 3), "slope=", round(slope, 3))
    for p in pts:
        if p["kind"] != "randk":
            print(p["layer"], p["kind"], p["k"], round(p["cos"], 2), round(p["retention"], 2))


if __name__ == "__main__":
    cosine_figure()
