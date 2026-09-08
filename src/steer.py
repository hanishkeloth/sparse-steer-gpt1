"""Demo: steer GPT-1's sentiment with a (sparse) steering vector.

    python src/steer.py --prompt "the interview went" --layer 7 --c 0.5 --k 12 --kind topk --sign +
"""
import argparse
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(__file__))
from common import generate, load_all, make_vector  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="the interview went")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--c", type=float, default=0.5, help="strength relative to mean residual norm at that block")
    ap.add_argument("--k", type=int, default=768, help="number of non-zero dimensions kept")
    ap.add_argument("--kind", choices=["topk", "signk", "randk"], default="topk")
    ap.add_argument("--sign", choices=["+", "-", "0"], default="+", help="+ positive, - negative, 0 unsteered")
    ap.add_argument("--n", type=int, default=3, help="samples")
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    model, tok = load_all()
    vecs = load_file(os.path.join(ROOT, "vectors", "gpt1_sentiment_vectors.safetensors"))
    v = vecs[f"layer_{a.layer}"]
    beta = a.c * float(vecs[f"resid_norm_{a.layer}"])
    import numpy as np
    u = make_vector(v, a.kind, a.k, rng=np.random.default_rng(a.seed))
    steer = None
    if a.sign != "0":
        steer = (a.layer, (beta if a.sign == "+" else -beta) * u)
        nz = int((u != 0).sum())
        print(f"steering block {a.layer} with {a.kind} k={nz} dims, beta={beta:.2f}, sign={a.sign}")
    outs = generate(model, tok, [a.prompt] * a.n, max_new=a.max_new, seed=a.seed, steer=steer)
    for o in outs:
        print(f"  {a.prompt} -> {tok.decode(o)}")


if __name__ == "__main__":
    main()
