"""Faithful re-implementation of the original GPT-1 text encoder (ftfy + spaCy + BPE with </w>).

Adapted from openai/finetune-transformer-lm/text_utils.py (MIT) so it runs on a modern spaCy
without a downloaded language model. Only the English rule-based tokenizer is used.
"""
import json
import re

import ftfy
from spacy.lang.en import English


def _get_pairs(word):
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


def text_standardize(text):
    text = text.replace("—", "-").replace("–", "-").replace("―", "-").replace("…", "...").replace("´", "'")
    text = re.sub(r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""", r" \1 ", text)
    text = re.sub(r"\s*\n\s*", " \n ", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


class GPT1Tokenizer:
    def __init__(self, encoder_path, bpe_path):
        self.nlp = English().tokenizer
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(bpe_path, encoding="utf-8").read().split("\n")[1:-1]
        self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        if out == "\n  </w>":
            out = "\n</w>"
        self.cache[token] = out
        return out

    def encode(self, text):
        doc = self.nlp(text_standardize(ftfy.fix_text(text)))
        ids = []
        for tok in doc:
            ids.extend(self.encoder.get(t, 0) for t in self.bpe(tok.text.lower()).split(" "))
        return ids

    def decode(self, ids):
        text = "".join(self.decoder.get(int(i), "<unk>") for i in ids)
        return text.replace("</w>", " ").strip()
