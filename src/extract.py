"""Extract per-layer sentiment steering vectors from GPT-1 by mean difference of contrastive pairs.

Writes vectors/gpt1_sentiment_vectors.safetensors with keys:
    layer_{L}          raw mean-difference vector at the output of block L  (768,)
    layer_{L}_std      per-dimension std of the residual at block L over the contrast set (768,)
    resid_norm_{L}     scalar mean ||h_L|| over the contrast set
plus vectors/meta.json.
"""
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(__file__))
from common import contrast_pairs, load_all, pad_batch, dump  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def last_token_states(model, tok, texts, bs=50):
    """Returns tensor (n_layers+1, N, d): hidden state at the last token after each block (index 0 = embeddings)."""
    chunks = []
    for i in range(0, len(texts), bs):
        ids, mask, pos = pad_batch(tok, texts[i:i + bs])
        hs = model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True).hidden_states
        chunks.append(torch.stack([h[:, -1, :] for h in hs]))  # right-aligned so last col is last token
    return torch.cat(chunks, 1)


def main():
    model, tok = load_all()
    pos_texts, neg_texts = contrast_pairs()
    hp = last_token_states(model, tok, pos_texts)
    hn = last_token_states(model, tok, neg_texts)
    n_layers = hp.shape[0] - 1
    out, meta = {}, {"n_pairs": len(pos_texts), "layers": {}}
    for L in range(n_layers):
        a, b = hp[L + 1], hn[L + 1]
        diff = (a - b).mean(0)
        allh = torch.cat([a, b], 0)
        out[f"layer_{L}"] = diff.contiguous()
        out[f"layer_{L}_std"] = allh.std(0).contiguous()
        out[f"resid_norm_{L}"] = allh.norm(dim=-1).mean().reshape(1)
        # separability: projection of individual pairs on the mean-diff direction
        u = diff / diff.norm()
        proj = ((a - b) @ u)
        meta["layers"][L] = {
            "diff_norm": float(diff.norm()),
            "resid_norm": float(allh.norm(dim=-1).mean()),
            "pair_projection_mean": float(proj.mean()),
            "pair_projection_frac_positive": float((proj > 0).float().mean()),
            "top16_dims": torch.topk(diff.abs(), 16).indices.tolist(),
        }
    os.makedirs(os.path.join(ROOT, "vectors"), exist_ok=True)
    save_file(out, os.path.join(ROOT, "vectors", "gpt1_sentiment_vectors.safetensors"),
              metadata={"model": "openai-gpt (GPT-1, 2018)", "method": "mean difference, last token",
                        "positive_minus_negative": "true"})
    dump(meta, os.path.join(ROOT, "vectors", "meta.json"))
    for L, m in meta["layers"].items():
        print(L, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "top16_dims"})


if __name__ == "__main__":
    main()
