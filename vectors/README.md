# GPT-1 sentiment steering vectors

`gpt1_sentiment_vectors.safetensors` (76 KB) — extracted from the original OpenAI GPT (2018) by mean
difference of the residual stream at the sentence-final period between 200 positive and 200 negative
template sentences (see `../src/extract.py`).

| key | shape | meaning |
|---|---|---|
| `layer_{L}` (L = 0..11) | (768,) | mean(h⁺) − mean(h⁻) at the output of block L; **positive − negative** |
| `layer_{L}_std` | (768,) | per-dimension std of the residual over the 400 extraction sentences |
| `resid_norm_{L}` | (1,) | mean ‖h_L‖ over the extraction set — multiply by `c` to get the steering strength β |

Usage (see `../src/steer.py`):

```python
from safetensors.torch import load_file
v = load_file("gpt1_sentiment_vectors.safetensors")
u = v["layer_9"] / v["layer_9"].norm()
beta = 0.25 * float(v["resid_norm_9"])
# add  beta * u  to the output of transformer.h[9] at every position (positive), or  -beta * u  (negative)
```

Blocks 0–5 carry almost no sentiment contrast at the period token (see paper, Fig. 1); use blocks 7–11.
Recommended: block 9 with c = 0.25, or block 11 with c = 0.5. `meta.json` lists per-block norms and the
top-16 coordinates.
