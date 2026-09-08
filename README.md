# sparse-steer-gpt1

**How few residual-stream dimensions does it take to steer a language model?**
A tiny, fully reproducible study of *sparse* and *one-bit* activation-steering vectors on the
original 2018 GPT (GPT-1, 117M) — a base model you can fetch with a single `git clone`, no model hub
required.

This repo ships:

* `vectors/gpt1_sentiment_vectors.safetensors` — open-weight, per-block sentiment steering vectors
  for GPT-1 (12 × 768 floats, plus per-dimension residual std and mean residual norms). 76 KB.
* `src/load_gpt1.py` + `src/gpt1_tokenizer.py` — a clean loader that turns OpenAI's original
  NumPy shards into a Hugging Face `OpenAIGPTLMHeadModel`, with a faithful port of the original
  ftfy + spaCy + BPE tokenizer (the HF tokenizer no longer loads these files on transformers ≥ 5).
* `src/run_experiments.py` — the full layer × strength × sparsity sweep (CPU-only, ~1.5 h on 2 cores).
* `paper/paper.md` / `paper/paper.pdf` — the write-up, with figures in `paper/figures/`.
* `results/` — every generation produced in the sweep (`raw_generations.jsonl`), the summary and
  the analysis used for the figures.

## Quick start

```bash
pip install -r requirements.txt
bash get_gpt1.sh ~/.cache/gpt1          # ~470 MB clone of openai/finetune-transformer-lm
export GPT1_DIR=~/.cache/gpt1/model

# unsteered
python src/steer.py --prompt "the interview went" --sign 0
# steer positive with the full vector at block 9
python src/steer.py --prompt "the interview went" --layer 9 --c 0.25 --sign +
# steer negative with a ONE-BIT vector that touches only 12 of 768 dimensions
python src/steer.py --prompt "the interview went" --layer 9 --c 0.25 --kind signk --k 12 --sign -
```

## Reproduce the paper

```bash
python src/extract.py          # contrastive extraction -> vectors/*.safetensors  (~1 min)
python src/run_experiments.py  # sweep -> results/summary.json                     (~1.5 h, CPU)
python src/analyze.py          # figures -> paper/figures, results/analysis.json
```

## Method in one paragraph

Sentiment vectors are the mean difference of GPT-1's residual stream between 200 positive and 200
negative template sentences (`"the movie was wonderful ."` vs `"the movie was terrible ."`), read at
the *shared* trailing period so the vector encodes sentiment context rather than the identity of the
adjective token. A vector is added (with a forward hook) to the output of one block at every position
while sampling 24 tokens from 24 neutral prompts. The **steering effect** is the difference in mean
VADER compound sentiment between the +vector and −vector runs; the **fluency cost** is the extra
per-token NLL of the generated text under the unsteered model. To sparsify, we keep the top-k
coordinates by magnitude (or only their *signs*, i.e. a one-bit vector), and rescale to the original
norm so the injected energy is constant; random-k supports are the control.

See `paper/paper.md` for results.

## Citation

```
@misc{sparsesteer2026,
  title  = {Sparse and One-Bit Steering Vectors: How Few Residual Dimensions Does It Take to Flip GPT-1's Sentiment?},
  author = {Hanish and Claude},
  year   = {2026},
  url    = {https://github.com/...}
}
```

MIT license. GPT-1 weights © OpenAI (MIT).
