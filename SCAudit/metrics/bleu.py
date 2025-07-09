"""Lightweight corpus‑level BLEU (N=4) with NumPy.
   Usage: score = corpus_bleu(candidates, list_of_reference_lists)
"""
from collections import Counter, defaultdict
from math import exp, log
from typing import List

MAX_N = 4  # 1‑ to 4‑gram


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(cands: List[List[str]], refs: List[List[List[str]]]) -> float:
    clip, cand_tot = [0] * MAX_N, [0] * MAX_N
    c_len = r_len = 0

    for cand_toks, ref_sets in zip(cands, refs):
        c_len += len(cand_toks)
        # length of ref closest to cand
        r_len += min(ref_sets, key=lambda r: abs(len(r) - len(cand_toks)), default=cand_toks).__len__()

        for n in range(1, MAX_N + 1):
            cand_ngrams = _ngram_counts(cand_toks, n)
            # max ref count per n‑gram
            ref_max = defaultdict(int)
            for ref in ref_sets:
                for g, ct in _ngram_counts(ref, n).items():
                    ref_max[g] = max(ref_max[g], ct)
            clip[n - 1] += sum(min(ct, ref_max[g]) for g, ct in cand_ngrams.items())
            cand_tot[n - 1] += sum(cand_ngrams.values())

    precisions = [clip[i] / cand_tot[i] if cand_tot[i] else 0 for i in range(MAX_N)]
    geo_mean = exp(sum(log(p or 1e-9) for p in precisions) / MAX_N)
    bp = 1.0 if c_len > r_len else exp(1.0 - r_len / max(c_len, 1))
    return bp * geo_mean
