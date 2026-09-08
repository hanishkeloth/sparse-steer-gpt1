"""Stage A: layer x strength sweep with the full steering vector.
Stage B: sparsity sweep (top-k, random-k, sign-k) at the selected layer(s).

Outputs results/raw_generations.jsonl (every generation) and results/summary.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, os.path.dirname(__file__))
from common import EVAL_PROMPTS, continuation_nll, dump, generate, load_all, make_vector  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
RAW = open(os.path.join(RES, "raw_generations.jsonl"), "a")

LAYERS_A = [1, 3, 5, 7, 9, 11]
C_LIST = [0.06, 0.12, 0.25, 0.5, 1.0]
K_LIST = [768, 384, 192, 96, 48, 24, 12, 6, 3, 1]
RANDOM_SEEDS = [0, 1, 2]
GEN_SEEDS_A = [0]
GEN_SEEDS_B = [0, 1]
MAX_NEW = 24

vader = SentimentIntensityAnalyzer()


def run_config(model, tok, tag, layer, vec, seeds, extra):
    """vec: scaled vector to add (or None for baseline). Returns per-direction summaries."""
    rows = []
    for seed in seeds:
        t0 = time.time()
        steer = None if vec is None else (layer, vec)
        outs = generate(model, tok, EVAL_PROMPTS, max_new=MAX_NEW, seed=seed, steer=steer)
        nll = continuation_nll(model, tok, EVAL_PROMPTS, outs)
        texts = [tok.decode(o) for o in outs]
        sent = [vader.polarity_scores(t)["compound"] for t in texts]
        for p, t, s, n in zip(EVAL_PROMPTS, texts, sent, nll):
            RAW.write(json.dumps({"tag": tag, "seed": seed, "prompt": p, "text": t, "vader": s, "nll": n, **extra}) + "\n")
        RAW.flush()
        rows.append({"seed": seed, "vader_mean": float(np.mean(sent)), "vader_pos_frac": float(np.mean([s > 0.05 for s in sent])),
                     "vader_neg_frac": float(np.mean([s < -0.05 for s in sent])), "nll_mean": float(np.mean(nll)),
                     "secs": time.time() - t0})
        print(f"{tag:40s} seed={seed} vader={rows[-1]['vader_mean']:+.3f} nll={rows[-1]['nll_mean']:.3f} ({rows[-1]['secs']:.0f}s)", flush=True)
    return {"vader_mean": float(np.mean([r["vader_mean"] for r in rows])),
            "vader_pos_frac": float(np.mean([r["vader_pos_frac"] for r in rows])),
            "vader_neg_frac": float(np.mean([r["vader_neg_frac"] for r in rows])),
            "nll_mean": float(np.mean([r["nll_mean"] for r in rows])), "per_seed": rows}


def bidirectional(model, tok, tag, layer, u, beta, seeds, extra):
    pos = run_config(model, tok, tag + "/+", layer, beta * u, seeds, {**extra, "direction": "+"})
    neg = run_config(model, tok, tag + "/-", layer, -beta * u, seeds, {**extra, "direction": "-"})
    return {"pos": pos, "neg": neg, "effect": pos["vader_mean"] - neg["vader_mean"],
            "nll_cost": 0.5 * (pos["nll_mean"] + neg["nll_mean"]), **extra}


def main():
    model, tok = load_all()
    torch.set_num_threads(os.cpu_count())
    vecs = load_file(os.path.join(ROOT, "vectors", "gpt1_sentiment_vectors.safetensors"))
    summary = {"baseline": None, "stage_a": [], "stage_b": [], "stage_a_selected": None}

    summary["baseline"] = run_config(model, tok, "baseline", None, None, GEN_SEEDS_B, {"kind": "baseline"})

    # ---- Stage A: layer x strength, full vector
    for L in LAYERS_A:
        v = vecs[f"layer_{L}"]
        u = v / v.norm()
        rn = float(vecs[f"resid_norm_{L}"])
        for c in C_LIST:
            r = bidirectional(model, tok, f"A/L{L}/c{c}", L, u, c * rn, GEN_SEEDS_A,
                              {"kind": "full", "layer": L, "c": c, "beta": c * rn, "k": 768})
            summary["stage_a"].append(r)
            dump(summary, os.path.join(RES, "summary.json"))

    # select: largest effect among configs whose fluency cost stays within +0.6 nats of baseline
    base_nll = summary["baseline"]["nll_mean"]
    ok = [r for r in summary["stage_a"] if r["nll_cost"] - base_nll < 0.6]
    best = max(ok or summary["stage_a"], key=lambda r: r["effect"])
    # second layer for robustness: best config among a different layer
    others = [r for r in ok if r["layer"] != best["layer"]]
    second = max(others, key=lambda r: r["effect"]) if others else None
    summary["stage_a_selected"] = {"best": {k: best[k] for k in ("layer", "c", "beta", "effect", "nll_cost")},
                                   "second": {k: second[k] for k in ("layer", "c", "beta", "effect", "nll_cost")} if second else None}
    print("SELECTED", summary["stage_a_selected"], flush=True)
    dump(summary, os.path.join(RES, "summary.json"))

    # ---- Stage B: sparsity sweep
    for i_sel, sel in enumerate([best] + ([second] if second else [])):
        L, beta = sel["layer"], sel["beta"]
        v = vecs[f"layer_{L}"]
        primary = i_sel == 0  # the second layer gets a lighter sweep (one seed, no random control)
        for k in K_LIST:
            for kind in ["topk", "signk"]:
                u = make_vector(v, kind, k)
                r = bidirectional(model, tok, f"B/L{L}/{kind}/k{k}", L, u, beta, GEN_SEEDS_B if primary else GEN_SEEDS_B[:1],
                                  {"kind": kind, "layer": L, "c": sel["c"], "beta": beta, "k": k})
                summary["stage_b"].append(r)
                dump(summary, os.path.join(RES, "summary.json"))
            if k < 768 and primary:
                for rs in RANDOM_SEEDS:
                    u = make_vector(v, "randk", k, rng=np.random.default_rng(1000 * rs + k))
                    r = bidirectional(model, tok, f"B/L{L}/randk{rs}/k{k}", L, u, beta, GEN_SEEDS_B[:1],
                                      {"kind": "randk", "layer": L, "c": sel["c"], "beta": beta, "k": k, "rand_seed": rs})
                    summary["stage_b"].append(r)
                    dump(summary, os.path.join(RES, "summary.json"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
