"""Exact (weighted) Chamfer scoring, IDF weights, and shared helpers."""
import json
import math
import os

import numpy as np
import torch

ROOT = os.environ.get("CERTLI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)


class Corpus:
    def __init__(self, model, dataset, dtype=np.float32):
        d = f"{ROOT}/emb/{model}/{dataset}"
        self.model, self.dataset = model, dataset
        self.E = np.load(f"{d}/doc_emb.npy").astype(dtype)
        self.tok = np.load(f"{d}/doc_tok.npy")
        self.off = np.load(f"{d}/doc_off.npy")
        self.dids = json.load(open(f"{d}/doc_ids.json"))
        self.Q = np.load(f"{d}/q_emb.npy").astype(np.float32)
        self.QT = np.load(f"{d}/q_tok.npy")
        self.qids = json.load(open(f"{d}/q_ids.json"))
        self.meta = json.load(open(f"{d}/meta.json"))
        self.N = len(self.dids)
        self.lens = np.diff(self.off)
        self.doc_of_tok = np.repeat(np.arange(self.N), self.lens)
        self.dim = self.E.shape[1]
        pos = {d: i for i, d in enumerate(self.dids)}
        # BEIR ignore_identical_ids: a query never retrieves the document with its own id
        self.self_doc = np.array([pos.get(q, -1) for q in self.qids])

    def idf_weights(self, special_weight=1.0):
        """Zero-shot Weighted Chamfer weights of [1]: BM25-style IDF over corpus documents.
        Special tokens (prefix, [CLS], [SEP], [MASK]) get `special_weight`; ids unseen in the
        corpus get 0, as in Archish et al. (AAAI 2026)."""
        N = self.N
        # document frequency per token id
        pairs = np.unique(np.stack([self.doc_of_tok, self.tok], 1), axis=0)
        df = np.bincount(pairs[:, 1], minlength=int(self.tok.max()) + 1)
        W = np.zeros(self.QT.shape, np.float32)
        specials = set(self.special_ids())
        for i in range(self.QT.shape[0]):
            for j in range(self.QT.shape[1]):
                t = int(self.QT[i, j])
                if t < 0:
                    continue
                if t in specials:
                    W[i, j] = special_weight
                elif t < len(df) and df[t] > 0:
                    n = df[t]
                    W[i, j] = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
                else:
                    W[i, j] = 0.0
        return W

    def special_ids(self):
        from transformers import AutoTokenizer

        tk = AutoTokenizer.from_pretrained(f"{ROOT}/assets/{self.model}")
        ids = set(tk.all_special_ids)
        for p in (self.meta.get("query_prefix"), self.meta.get("document_prefix")):
            if p:
                ids.add(tk.convert_tokens_to_ids(p))
        return sorted(ids)


def seg_max(A, starts, axis=0):
    """Segment max along an axis given segment start indices (segments non-empty)."""
    return np.maximum.reduceat(A, starts, axis=axis)


def chamfer_all(Qtok, W, E, off, q_batch=8, tok_chunk=262144, shift=None):
    """Exact weighted Chamfer S[q, D] = sum_i w_i max_j <q_i, e_j> for all queries x docs.
    Qtok: [nq, Lq, d]; W: [nq, Lq]; E: [T, d]; off: [N+1]. Returns float64 [nq, N].
    Also returns per-token maxima M [nq, Lq, N] if requested via shift=None (not stored)."""
    nq, Lq, d = Qtok.shape
    N = len(off) - 1
    out = np.zeros((nq, N), np.float64)
    Et = torch.from_numpy(E)
    # chunk boundaries aligned to documents
    bounds = [0]
    for i in range(1, N + 1):
        if off[i] - off[bounds[-1]] >= tok_chunk or i == N:
            bounds.append(i)
    for b0 in range(0, nq, q_batch):
        qb = Qtok[b0 : b0 + q_batch]
        B = qb.shape[0]
        qt = torch.from_numpy(qb.reshape(B * Lq, d))
        wb = W[b0 : b0 + q_batch]
        for c0, c1 in zip(bounds[:-1], bounds[1:]):
            t0, t1 = off[c0], off[c1]
            sims = torch.matmul(qt, Et[t0:t1].T).numpy()  # [B*Lq, t]
            m = seg_max(sims, off[c0:c1] - t0, axis=1)  # [B*Lq, docs]
            m = m.reshape(B, Lq, c1 - c0).astype(np.float64)
            out[b0 : b0 + B, c0:c1] = np.einsum("bld,bl->bd", m, wb)
    return out


def topk_ids(scores, k, exclude=None):
    s = scores.copy()
    if exclude is not None and exclude >= 0:
        s[exclude] = -np.inf
    idx = np.argpartition(-s, k)[:k]
    return idx[np.argsort(-s[idx], kind="stable")]
