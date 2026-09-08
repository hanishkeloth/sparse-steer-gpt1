"""Shared helpers: data, batched generation with left padding, steering hooks, scoring."""
import json
import os
import random

import numpy as np
import torch

from load_gpt1 import load_gpt1

CONTEXTS = [
    "the movie was", "the food at this restaurant was", "the hotel room was", "this book is",
    "the concert last night was", "my new phone is", "the service here was", "the weather today is",
    "the teacher was", "the game was", "our vacation was", "the coffee tasted", "the play was",
    "the customer support was", "the neighborhood is", "the album sounds", "the flight was",
    "the presentation was", "the party was", "the car drives",
]
POS_ADJ = ["wonderful", "excellent", "fantastic", "delightful", "amazing", "great", "lovely", "superb",
           "brilliant", "perfect"]
NEG_ADJ = ["terrible", "awful", "horrible", "dreadful", "disappointing", "bad", "miserable", "atrocious",
           "boring", "useless"]

EVAL_PROMPTS = [
    "i walked into the store and", "when i got home , i", "the meeting this morning",
    "she looked at the report and said", "after the long drive we", "the first day of school was",
    "my brother called to tell me", "the new apartment", "we sat down for dinner and",
    "the doctor came in and", "on saturday we went to", "the package arrived and",
    "he opened the letter and", "the city at night is", "the train ride to work",
    "i started the new job and", "the old house on the corner", "the kids came back from the trip and",
    "the reviews of the show said", "at the end of the day , i felt", "the garden this year",
    "the museum exhibit was", "our neighbors are", "the interview went",
]


def contrast_pairs(readout=" ."):
    """Contrastive pairs. A shared trailing token (a period) is appended so that the state we read out
    encodes the sentiment *context* rather than the identity of the adjective token itself."""
    pos, neg = [], []
    for ctx in CONTEXTS:
        for p, n in zip(POS_ADJ, NEG_ADJ):
            pos.append(f"{ctx} {p}{readout}")
            neg.append(f"{ctx} {n}{readout}")
    return pos, neg


def pad_batch(tok, texts, pad_id=0):
    ids = [tok.encode(t) for t in texts]
    L = max(len(x) for x in ids)
    input_ids = torch.full((len(ids), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, L - len(x):] = torch.tensor(x)
        mask[i, L - len(x):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return input_ids, mask, pos


class Steer:
    """Adds `vec` (already scaled) to the output of transformer block `layer` at every position."""

    def __init__(self, model, layer, vec):
        self.vec = vec
        self.h = model.transformer.h[layer].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        out[0] = out[0] + self.vec.to(out[0].dtype)
        return out

    def remove(self):
        self.h.remove()


@torch.no_grad()
def generate(model, tok, prompts, max_new=24, temperature=0.8, top_p=0.9, seed=0, steer=None):
    """Batched nucleus sampling without KV cache (GPT-1 in HF has none). Returns list of token lists."""
    g = torch.Generator().manual_seed(seed)
    input_ids, mask, pos = pad_batch(tok, prompts)
    out_tokens = [[] for _ in prompts]
    hook = Steer(model, *steer) if steer is not None else None
    try:
        for _ in range(max_new):
            logits = model(input_ids=input_ids, attention_mask=mask, position_ids=pos).logits[:, -1, :]
            logits = logits / temperature
            probs = torch.softmax(logits, -1)
            sp, si = probs.sort(-1, descending=True)
            cum = sp.cumsum(-1)
            keep = (cum - sp) < top_p
            sp = sp * keep
            sp = sp / sp.sum(-1, keepdim=True)
            choice = torch.multinomial(sp, 1, generator=g)
            nxt = si.gather(-1, choice)
            for i in range(len(prompts)):
                out_tokens[i].append(int(nxt[i]))
            input_ids = torch.cat([input_ids, nxt], 1)
            mask = torch.cat([mask, torch.ones_like(nxt)], 1)
            pos = torch.cat([pos, pos[:, -1:] + 1], 1)
    finally:
        if hook is not None:
            hook.remove()
    return out_tokens


@torch.no_grad()
def continuation_nll(model, tok, prompts, continuations):
    """Mean per-token NLL of each continuation under the (unsteered) model, given its prompt."""
    texts_ids = []
    plens = []
    for p, c in zip(prompts, continuations):
        pid = tok.encode(p)
        texts_ids.append(pid + c)
        plens.append(len(pid))
    L = max(len(x) for x in texts_ids)
    input_ids = torch.zeros((len(texts_ids), L), dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for i, x in enumerate(texts_ids):
        input_ids[i, L - len(x):] = torch.tensor(x)
        mask[i, L - len(x):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=input_ids, attention_mask=mask, position_ids=pos).logits
    logp = torch.log_softmax(logits[:, :-1], -1)
    tgt = input_ids[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    nlls = []
    for i, x in enumerate(texts_ids):
        start = L - len(x) + plens[i] - 1  # index into tok_lp for first continuation token
        nlls.append(float(-tok_lp[i, start:].mean()))
    return nlls


def load_all(model_dir=None):
    model_dir = model_dir or os.environ.get("GPT1_DIR", "/tmp/gpt1/model")
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    return load_gpt1(model_dir)


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)


def make_vector(v, kind, k, rng=None):
    """Return a unit-norm steering direction built from raw mean-difference vector v."""
    d = v.numel()
    if kind == "topk":
        idx = torch.topk(v.abs(), k).indices
        u = torch.zeros_like(v)
        u[idx] = v[idx]
    elif kind == "randk":
        idx = torch.tensor(rng.choice(d, size=k, replace=False))
        u = torch.zeros_like(v)
        u[idx] = v[idx]
    elif kind == "signk":  # 1-bit: only the sign of the top-k coordinates survives
        idx = torch.topk(v.abs(), k).indices
        u = torch.zeros_like(v)
        u[idx] = torch.sign(v[idx])
    else:
        raise ValueError(kind)
    return u / u.norm()
