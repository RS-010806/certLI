"""E3/E4/E6/E8/E9: certification cost of CertLI codes.

For each code we report bytes/doc relative to fp16 full token sets, bound validity, the number of
full-token-set reads that exact top-k certification needs (read-optimal, = #{D : U_D >= s_k}),
point-estimate quality, and the risk-controlled (Learn-then-Test) variant.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from certify import cert_stats, depth_sweep, shrink_sweep  # noqa: E402
from codec import build_lrs, build_pool, build_rq  # noqa: E402
from core import ROOT, Corpus  # noqa: E402
from data import evaluate_scores, load_beir  # noqa: E402

LAMBDAS = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.0]
DEPTHS = [10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 75, 90, 100, 125, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 60000]


def summarize(code, S_sh, U, L, Sh, k, C, qrels, const, want_sweeps):
    st = cert_stats(S_sh, U, L, Sh, k, C.self_doc)
    fr = np.array(st["frac_read"])
    nr = np.array(st["n_read"])
    ev = evaluate_scores(Sh + const[:, None], C.qids, C.dids, qrels)
    row = dict(code=code.name, params=code.params,
               bytes_ratio=float(code.bytes_of.sum() / code.full_bytes.sum()),
               bytes_per_doc=float(code.bytes_of.mean()), full_bytes_per_doc=float(code.full_bytes.mean()),
               mean_k=float(np.mean(code.k_of)), mean_clusters=float(np.mean(code.nc_of)),
               mean_verbatim=float(np.mean(code.nv_of)), mean_tokens=float(np.mean(code.m_of)),
               violations_lo=st["viol_lo"], violations_hi=st["viol_hi"], pairs=st["pairs"],
               reads_mean=float(nr.mean()), reads_median=float(np.median(nr)),
               frac_read_mean=float(fr.mean()), frac_read_median=float(np.median(fr)),
               frac_read_p90=float(np.quantile(fr, 0.9)), frac_read_max=float(fr.max()),
               frac_queries_read_le_5pct=float((fr <= 0.05).mean()),
               frac_queries_read_le_20pct=float((fr <= 0.20).mean()),
               width_mean=float(np.mean(st["width"])), slack_up_mean=float(np.mean(st["slack_up"])),
               err_hat_mean=float(np.mean(st["err_hat"])), s_k_mean=float(np.mean(st["s_k"])),
               gap_k_mean=float(np.mean(st["gap_k"])),
               approx_recall_at_k=float(np.mean(st["approx_recall"])), approx_ndcg10=ev["ndcg@10"])
    extra = dict(n_read=nr.tolist(), frac_read=fr.tolist(), gap_k=st["gap_k"], s_k=st["s_k"])
    if want_sweeps:
        reads, rec = shrink_sweep(S_sh, U, Sh, k, C.self_doc, LAMBDAS)
        extra["lambdas"] = LAMBDAS
        extra["shrink_reads"] = reads.tolist()
        extra["shrink_recall"] = rec.tolist()
        depths = [d for d in DEPTHS if d <= C.N] + ([C.N] if C.N not in DEPTHS else [])
        drec = depth_sweep(S_sh, Sh, k, C.self_doc, depths)
        extra["depths"] = depths
        extra["depth_recall"] = drec.tolist()
    return row, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dataset")
    ap.add_argument("--weights", default="w1")
    ap.add_argument("--suite", default="main")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nosweep", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    C = Corpus(a.model, a.dataset)
    _, _, qrels = load_beir(a.dataset)
    d = f"{ROOT}/emb/{a.model}/{a.dataset}"
    S = np.load(f"{d}/S_{a.weights}.npy").astype(np.float64)
    W = np.load(f"{d}/W_{a.weights}.npy")
    mu = C.E.astype(np.float64).mean(0)
    sigma = float(np.sqrt(((C.E.astype(np.float64) - mu) ** 2).sum(1).mean()))
    sigma_raw = float(np.sqrt((C.E.astype(np.float64) ** 2).sum(1).mean()))
    caches = {True: {}, False: {}}

    configs = []
    if a.suite == "full":
        for t in (0.3, 0.5, 0.7):
            for e in (0.25, 0.45, 0.7):
                configs.append(("lrs", dict(tau=t, eps=e, shift=True), (t, e) in {(0.5, 0.45), (0.7, 0.7)}))
        configs.append(("lrs", dict(tau=0.5, eps=0.45, shift=False), False))
        configs.append(("lrs", dict(tau=0.5, eps=0.45, shift=True, local_radii=False), False))
        for e in (0.45, 0.7):
            configs.append(("pool", dict(eps=e, shift=True), False))
        for nb in (1, 2, 4):
            configs.append(("rq", dict(nbits=nb, shift=True), nb == 2))
    elif a.suite == "main":  # reduced grid used for the second dataset / second model
        for t, e in ((0.3, 0.25), (0.5, 0.25), (0.5, 0.45), (0.7, 0.7)):
            configs.append(("lrs", dict(tau=t, eps=e, shift=True), (t, e) in {(0.5, 0.45), (0.7, 0.7)}))
        configs.append(("lrs", dict(tau=0.5, eps=0.45, shift=False), False))
        configs.append(("lrs", dict(tau=0.5, eps=0.45, shift=True, local_radii=False), False))
        configs.append(("pool", dict(eps=0.45, shift=True), False))
        for nb in (1, 2, 4):
            configs.append(("rq", dict(nbits=nb, shift=True), nb == 2))
    elif a.suite == "small":
        configs.append(("lrs", dict(tau=0.5, eps=0.45, shift=True), True))
        configs.append(("lrs", dict(tau=0.5, eps=0.45, shift=False), False))
        configs.append(("rq", dict(nbits=2, shift=True), True))
    elif a.suite == "bonly":
        configs.append(("lrs", dict(tau=0.5, eps=0.45, shift=True), True))
    elif a.suite == "rq2":
        configs.append(("rq", dict(nbits=2, shift=True), True))
    elif a.suite == "shiftabl":
        for t, e in ((0.3, 0.25), (0.5, 0.45)):
            configs.append(("lrs", dict(tau=t, eps=e, shift=False, scale="shifted"), False))
    elif a.suite == "rq":
        for nb in (1, 2, 4):
            configs.append(("rq", dict(nbits=nb, shift=True), nb == 2))
    elif a.suite == "sweeps":
        configs.append(("lrs", dict(tau=0.5, eps=0.45, shift=True), True))
        configs.append(("lrs", dict(tau=0.7, eps=0.7, shift=True), True))
        configs.append(("rq", dict(nbits=2, shift=True), True))
    elif a.suite == "idf":
        configs.append(("lrs", dict(tau=0.5, eps=0.45, shift=True), True))
        configs.append(("rq", dict(nbits=2, shift=True), True))

    rows, extras = [], []
    code = None
    for kind, p, sweep in configs:
        code = None
        import gc; gc.collect()
        tb = time.time()
        if kind == "lrs":
            shift = p.get("shift", True)
            sc = sigma if (shift or p.get("scale") == "shifted") else sigma_raw
            code = build_lrs(C.E, C.off, tau=p["tau"] * sc, eps=p["eps"] * sc, shift=shift,
                             mu=mu if shift else np.zeros_like(mu), local_radii=p.get("local_radii", True),
                             eig_cache=caches[shift])
        elif kind == "pool":
            code = build_pool(C.E, C.off, eps=p["eps"] * sigma, shift=True, mu=mu)
        else:
            caches[True].clear(); caches[False].clear()
            code = build_rq(C.E, C.off, nbits=p["nbits"], n_centroids=4096, shift=True, mu=mu)
        tbuild = time.time() - tb
        code.params.update({"rel": p})
        tq = time.time()
        U, L, Sh = code.bounds(C.Q, W, q_batch=1)
        tbound = time.time() - tq
        const = np.einsum("qld,ql,d->q", C.Q.astype(np.float64), W.astype(np.float64), code.mu.astype(np.float64))
        S_sh = S - const[:, None]
        row, extra = summarize(code, S_sh, U, L, Sh, a.k, C, qrels, const, sweep and not a.nosweep)
        if sweep:
            os.makedirs(f"{ROOT}/results/bounds", exist_ok=True)
            np.savez_compressed(f"{ROOT}/results/bounds/{a.model}__{a.dataset}__{a.weights}__{kind}__{len(rows)}.npz",
                                S_sh=S_sh.astype(np.float32), U=U.astype(np.float32), L=L.astype(np.float32),
                                Sh=Sh.astype(np.float32), self_doc=C.self_doc, rel=json.dumps(p))
        row.update(kind=kind, rel=p, build_s=tbuild, bound_s=tbound, sweep=sweep)
        if kind == "rq":
            row["rq_mean_err"] = code.mean_err
            row["rq_codebook_bytes"] = code.centroid_bytes
        rows.append(row)
        extras.append(extra)
        print(f"[{a.model}/{a.dataset}/{a.weights}] {kind} {p} bytes {row['bytes_ratio']:.3f} "
              f"read mean {row['frac_read_mean']:.3f} med {row['frac_read_median']:.3f} "
              f"viol {row['violations_lo']},{row['violations_hi']} apxR {row['approx_recall_at_k']:.3f} "
              f"apx_nDCG {row['approx_ndcg10']:.4f} | build {tbuild:.0f}s bound {tbound:.0f}s", flush=True)
        out = dict(model=a.model, dataset=a.dataset, weights=a.weights, k=a.k, sigma=sigma,
                   sigma_raw=sigma_raw, n_docs=C.N, n_queries=len(C.qids), rows=rows)
        os.makedirs(f"{ROOT}/results/cert", exist_ok=True)
        tag = f"{a.model}__{a.dataset}__{a.weights}__{a.suite}"
        json.dump(out, open(f"{ROOT}/results/cert/{tag}.json", "w"), indent=1)
        json.dump(extras, open(f"{ROOT}/results/cert/{tag}__extra.json", "w"))
    print("total", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
