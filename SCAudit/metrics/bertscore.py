"""Asynchronous batch BERTScore using HuggingFace Transformers."""
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from typing import List

_MODEL_CACHE = {}

def _get_model(name: str):
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).eval()
    _MODEL_CACHE[name] = (tok, mdl)
    return _MODEL_CACHE[name]

async def bertscore(cands: List[str], refs: List[str], model_name="microsoft/deberta-base") -> float:
    tok, mdl = _get_model(model_name)
    with torch.no_grad():
        c_enc = tok(cands, padding=True, truncation=True, return_tensors="pt")
        r_enc = tok(refs,  padding=True, truncation=True, return_tensors="pt")
        c_emb = mdl(**c_enc).last_hidden_state  # [B, Lc, d]
        r_emb = mdl(**r_enc).last_hidden_state  # [B, Lr, d]
        # Max‑token alignment
        sim = cosine_similarity(c_emb.unsqueeze(2), r_emb.unsqueeze(1), dim=-1)
        prec = sim.max(2).values.mean(1)  # best ref match per cand token
        rec  = sim.max(1).values.mean(1)  # best cand match per ref token
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return f1.mean().item()
