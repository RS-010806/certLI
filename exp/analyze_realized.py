"""Doc-level view of the slack: for every (query, document) pair, the realised position of the exact
score inside its certified interval, r = (S - Shat) / (U - Shat). A shrink factor lambda is safe for a
pair exactly when r <= lambda; the LTT-selected lambda should sit near the upper tail of r for the
documents that compete for the top-k."""
import glob
import json
import os

import numpy as np

ROOT = os.environ.get("CERTLI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = {}
for p in sorted(glob.glob(f"{ROOT}/results/bounds/*.npz")):
    z = np.load(p)
    S, U, Sh, sd = z["S_sh"].astype(np.float64), z["U"].astype(np.float64), z["Sh"].astype(np.float64), z["self_doc"]
    nq, N = S.shape
    r_all, r_top = [], []
    gaps, widths = [], []
    for qi in range(nq):
        s, u, sh = S[qi].copy(), U[qi].copy(), Sh[qi].copy()
        if sd[qi] >= 0:
            s[sd[qi]] = np.nan
        ok = np.isfinite(s)
        w = u - sh
        r = (s - sh) / np.maximum(w, 1e-9)
        r_all.append(r[ok])
        srt = np.sort(s[ok])[::-1]
        sk = srt[9]
        top = ok & (s >= sk)
        r_top.append(r[top])
        gaps.append(srt[9] - srt[10])
        widths.append(np.mean(u[ok] - s[ok]))
    r_all = np.concatenate(r_all); r_top = np.concatenate(r_top)
    q = lambda x, a: float(np.quantile(x, a))
    key = os.path.basename(p)[:-4]
    out[key] = dict(rel=str(z["rel"]), r_all_median=q(r_all, 0.5), r_all_p99=q(r_all, 0.99), r_all_p999=q(r_all, 0.999),
                    r_all_max=float(r_all.max()), r_top_median=q(r_top, 0.5), r_top_p90=q(r_top, 0.9),
                    r_top_p99=q(r_top, 0.99), r_top_max=float(r_top.max()),
                    gap_k_median=float(np.median(gaps)), gap_k_mean=float(np.mean(gaps)),
                    upper_slack_mean=float(np.mean(widths)), s_scale=float(np.mean(np.sort(S, 1)[:, -10])))
    print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[key].items()})
os.makedirs(f"{ROOT}/results/realized", exist_ok=True)
json.dump(out, open(f"{ROOT}/results/realized/all.json", "w"), indent=1)

# ---- histograms for the figure (LRS tau=0.5, eps=0.45 configs only)
stop = json.load(open(f"{ROOT}/results/stop/all.json")) if os.path.exists(f"{ROOT}/results/stop/all.json") else {}
MS = {"colbertv2": "ColBERTv2", "answerai-small": "AnswerAI-small"}
DS = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}
hist = {}
edges = np.linspace(-0.1, 1.0, 45)
for p in sorted(glob.glob(f"{ROOT}/results/bounds/*__lrs__*.npz")):
    z = np.load(p)
    rel = str(z["rel"])
    if '"tau": 0.5' not in rel or '"eps": 0.45' not in rel:
        continue
    key = os.path.basename(p)[:-4]
    S, U, Sh, sd = z["S_sh"].astype(np.float64), z["U"].astype(np.float64), z["Sh"].astype(np.float64), z["self_doc"]
    ra, rt = [], []
    for qi in range(S.shape[0]):
        s, u, sh = S[qi].copy(), U[qi].copy(), Sh[qi].copy()
        if sd[qi] >= 0:
            s[sd[qi]] = np.nan
        ok = np.isfinite(s)
        r = (s - sh) / np.maximum(u - sh, 1e-9)
        sk = np.sort(s[ok])[::-1][9]
        ra.append(r[ok]); rt.append(r[ok & (s >= sk)])
    ra = np.clip(np.concatenate(ra), -0.0999, 0.9999); rt = np.clip(np.concatenate(rt), -0.0999, 0.9999)
    lam = None
    if key in stop:
        g = stop[key]["shrink"]["grid"]; lam = g[int(stop[key]["shrink"]["ltt05"]["param_index_median"])]
    m, d = key.split("__")[:2]
    hist[key] = dict(edges=edges.tolist(), all=np.histogram(ra, edges)[0].tolist(), top=np.histogram(rt, edges)[0].tolist(),
                     lam=lam, title=f"{MS.get(m, m)} / {DS.get(d, d)}")
json.dump(hist, open(f"{ROOT}/results/realized/hist.json", "w"))
print("hist keys", list(hist))
