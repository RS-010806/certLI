"""Encode BEIR corpora and queries with PyLate ColBERT models.

Stores per (model, dataset):
  doc_emb.npy  float16 [T, dim]   L2-normalised token vectors (skiplist and padding removed)
  doc_tok.npy  int32   [T]        token ids aligned with doc_emb
  doc_off.npy  int64   [N+1]      offsets into doc_emb per document
  doc_ids.json
  q_emb.npy    float32 [Q, Lq, dim]  query vectors incl. [MASK] expansion
  q_tok.npy    int32   [Q, Lq]
  q_ids.json
  meta.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import load_beir  # noqa: E402

ROOT = os.environ.get("CERTLI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIRS = {
    "colbertv2": f"{ROOT}/assets/colbertv2",
    "answerai-small": f"{ROOT}/assets/answerai-small",
    "gte-moderncolbert": f"{ROOT}/assets/gte-moderncolbert",
}


def load_model(key, doc_len=300, q_len=32):
    from pylate import models

    return models.ColBERT(
        model_name_or_path=MODEL_DIRS[key],
        document_length=doc_len,
        query_length=q_len,
        device="cpu",
    )


def _filter(out):
    embs, toks = [], []
    for e, ids, m in zip(out["token_embeddings"], out["input_ids"], out["masks"]):
        e = np.asarray(e, dtype=np.float32)
        ids = np.asarray(ids)
        m = np.asarray(m).astype(bool)
        e = e[m]
        e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)  # re-normalise in fp32
        embs.append(e)
        toks.append(ids[m].astype(np.int32))
    return embs, toks


def run_encode(model_key, dataset, bf16=False, doc_len=300, q_len=32, chunk=1500, bs=32, limit=None):
    torch.set_num_threads(2)
    out_dir = f"{ROOT}/emb/{model_key}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)
    corpus, queries, qrels = load_beir(dataset)
    dids = list(corpus.keys())
    if limit:
        dids = dids[:limit]
    qids = list(queries.keys())
    model = load_model(model_key, doc_len, q_len)
    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else torch.no_grad()

    # queries
    t0 = time.time()
    with ctx, torch.no_grad():
        qo = model.encode([queries[q] for q in qids], is_query=True, batch_size=64,
                          output_value=None, show_progress_bar=False)
    q_embs, q_toks = _filter(qo)
    lq = max(len(t) for t in q_toks)
    dim = q_embs[0].shape[1]
    Q = np.zeros((len(qids), lq, dim), np.float32)
    QT = np.full((len(qids), lq), -1, np.int32)
    for i, (e, t) in enumerate(zip(q_embs, q_toks)):
        Q[i, : len(t)] = e
        QT[i, : len(t)] = t
    np.save(f"{out_dir}/q_emb.npy", Q)
    np.save(f"{out_dir}/q_tok.npy", QT)
    json.dump(qids, open(f"{out_dir}/q_ids.json", "w"))
    tq = time.time() - t0

    # documents, chunked with resume
    parts = []
    t0 = time.time()
    n_tok = 0
    for ci, s in enumerate(range(0, len(dids), chunk)):
        p = f"{out_dir}/part_{ci:04d}.npz"
        if os.path.exists(p):
            parts.append(p)
            continue
        texts = [corpus[d] for d in dids[s : s + chunk]]
        with ctx, torch.no_grad():
            do = model.encode(texts, is_query=False, batch_size=bs, output_value=None,
                              show_progress_bar=False)
        embs, toks = _filter(do)
        lens = np.array([len(t) for t in toks], np.int64)
        np.savez(p, emb=np.concatenate(embs).astype(np.float16), tok=np.concatenate(toks), lens=lens)
        parts.append(p)
        n_tok += int(lens.sum())
        el = time.time() - t0
        print(f"[{model_key}/{dataset}] chunk {ci} docs {s+len(texts)}/{len(dids)} "
              f"elapsed {el:.0f}s ({(s+len(texts))/max(el,1e-9):.1f} docs/s)", flush=True)
    E, T, L = [], [], []
    for p in parts:
        z = np.load(p)
        E.append(z["emb"]); T.append(z["tok"]); L.append(z["lens"])
    E = np.concatenate(E); T = np.concatenate(T); L = np.concatenate(L)
    off = np.zeros(len(L) + 1, np.int64)
    off[1:] = np.cumsum(L)
    np.save(f"{out_dir}/doc_emb.npy", E)
    np.save(f"{out_dir}/doc_tok.npy", T)
    np.save(f"{out_dir}/doc_off.npy", off)
    json.dump(dids, open(f"{out_dir}/doc_ids.json", "w"))
    meta = dict(model=model_key, dataset=dataset, bf16=bf16, doc_len=doc_len, q_len=q_len,
                n_docs=len(dids), n_queries=len(qids), n_tokens=int(off[-1]), dim=int(E.shape[1]),
                query_prefix=model.query_prefix, document_prefix=model.document_prefix,
                do_query_expansion=model.do_query_expansion,
                attend_to_expansion_tokens=model.attend_to_expansion_tokens,
                mask_token_id=model.tokenizer.mask_token_id, query_time_s=tq,
                doc_time_s=time.time() - t0)
    json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=1)
    for p in parts:
        os.remove(p)
    print("DONE", json.dumps(meta), flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--doc_len", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for ds in a.datasets:
        run_encode(a.model, ds, bf16=a.bf16, doc_len=a.doc_len, limit=a.limit)
