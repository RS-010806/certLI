"""BEIR loading and IR metrics (trec_eval conventions)."""
import csv
import gzip
import io
import json
import os
import zipfile

import numpy as np

ASSETS = os.path.join(os.environ.get("CERTLI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")


def _read_jsonl(fh):
    out = []
    for line in fh:
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_beir(name, split="test"):
    """Return corpus (dict id -> text), queries (dict id -> text), qrels (dict qid -> {did: rel})."""
    zpath = os.path.join(ASSETS, f"beir_{name}.zip")
    if os.path.exists(zpath):
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()

            def find(suffix):
                cand = [n for n in names if n.endswith(suffix)]
                if not cand:
                    raise FileNotFoundError(suffix)
                return cand[0]

            with z.open(find("corpus.jsonl")) as f:
                corpus_rows = _read_jsonl(io.TextIOWrapper(f, encoding="utf-8"))
            with z.open(find("queries.jsonl")) as f:
                query_rows = _read_jsonl(io.TextIOWrapper(f, encoding="utf-8"))
            with z.open(find(f"qrels/{split}.tsv")) as f:
                qrel_lines = io.TextIOWrapper(f, encoding="utf-8").read().splitlines()
    else:
        base = os.path.join(ASSETS, "beir_hf", name)
        with gzip.open(os.path.join(base, "corpus.jsonl.gz"), "rt", encoding="utf-8") as f:
            corpus_rows = _read_jsonl(f)
        with gzip.open(os.path.join(base, "queries.jsonl.gz"), "rt", encoding="utf-8") as f:
            query_rows = _read_jsonl(f)
        with open(os.path.join(base, f"{split}.tsv"), encoding="utf-8") as f:
            qrel_lines = f.read().splitlines()

    qrels = {}
    reader = csv.reader(qrel_lines, delimiter="\t")
    header = next(reader)
    for row in reader:
        if len(row) < 3:
            continue
        qid, did, rel = row[0], row[1], int(float(row[2]))
        qrels.setdefault(qid, {})[did] = rel

    corpus = {}
    for r in corpus_rows:
        did = str(r["_id"])
        title = (r.get("title") or "").strip()
        text = (r.get("text") or "").strip()
        corpus[did] = (title + " " + text).strip() if title else text
    queries = {str(r["_id"]): r["text"] for r in query_rows if str(r["_id"]) in qrels}
    qrels = {q: v for q, v in qrels.items() if q in queries}
    return corpus, queries, qrels


# ---------------------------------------------------------------- metrics
def ndcg_at_k(ranked_ids, rels, k=10):
    """trec_eval ndcg_cut: linear gain, log2 discount, ideal from all judged rel>0."""
    dcg = 0.0
    for i, d in enumerate(ranked_ids[:k]):
        g = rels.get(d, 0)
        if g > 0:
            dcg += g / np.log2(i + 2)
    ideal = sorted([r for r in rels.values() if r > 0], reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, rels, k=100):
    pos = {d for d, r in rels.items() if r > 0}
    if not pos:
        return 0.0
    return len(pos.intersection(ranked_ids[:k])) / len(pos)


def mrr_at_k(ranked_ids, rels, k=10):
    for i, d in enumerate(ranked_ids[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_scores(score_matrix, qids, dids, qrels, ignore_identical=True, ks=(10, 100)):
    """score_matrix: (nq, nd) float. Returns mean nDCG@10, Recall@100, MRR@10 and per-query lists."""
    nd10, r100, m10 = [], [], []
    did_arr = np.asarray(dids)
    pos_of = {d: i for i, d in enumerate(dids)}
    for qi, q in enumerate(qids):
        s = score_matrix[qi].astype(np.float64).copy()
        if ignore_identical and q in pos_of:
            s[pos_of[q]] = -np.inf
        top = np.argpartition(-s, 100)[:100]
        top = top[np.argsort(-s[top], kind="stable")]
        ranked = list(did_arr[top])
        rels = qrels[q]
        nd10.append(ndcg_at_k(ranked, rels, 10))
        r100.append(recall_at_k(ranked, rels, 100))
        m10.append(mrr_at_k(ranked, rels, 10))
    return {
        "ndcg@10": float(np.mean(nd10)),
        "recall@100": float(np.mean(r100)),
        "mrr@10": float(np.mean(m10)),
        "per_query_ndcg@10": nd10,
    }
