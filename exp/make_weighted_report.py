"""Table for the one-page note on weighted EigenLI (report/weighted_eigenli.tex). Every number is read from results/.

Paired two-sided sign-flip permutation test over queries (20,000 permutations, seed 0) for weighted vs plain EigenLI.
"""
import json
import os

import numpy as np

ROOT = os.environ.get("CERTLI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = f"{ROOT}/tables"
MS = {"colbertv2": "ColBERTv2", "answerai-small": "AnswerAI-small"}
DS = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna"}
PAIRS = [(m, d) for m in ("colbertv2", "answerai-small") for d in ("scifact", "nfcorpus")]
rng = np.random.default_rng(0)


def perm_p(diff, n=20000):
    diff = np.asarray(diff, float)
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(diff)))
    return float((1 + (np.abs((signs * diff).mean(1)) >= obs - 1e-15).sum()) / (n + 1))


def stars(p):
    return "$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "")


def signed(dd):
    return f"{dd:+.1f}".replace("-", "$-$")


rows, out = [], {}
for m, d in PAIRS + [("answerai-small", "arguana")]:
    fp = f"{ROOT}/results/eigen_weighted/{m}__{d}.json"
    if not os.path.exists(fp):
        continue
    r = json.load(open(fp))
    R = {(x["weights"], x["shift"], x["k"]): x for x in r["rows"]}
    ex = round(100 * r["exact"]["uniform"], 1)
    cells, rec = [], {}
    for k in (16, 32):
        a = np.array(R[("uniform", False, k)]["per_query_ndcg10"]); b = np.array(R[("idf", False, k)]["per_query_ndcg10"])
        p = perm_p(b - a)
        va, vb = round(100 * a.mean(), 1), round(100 * b.mean(), 1)
        cells += [f"{va:.1f}", f"\\textbf{{{vb:.1f} ({signed(vb - va)})}}{stars(p)}" if vb > va else f"{vb:.1f} ({signed(vb - va)}){stars(p)}"]
        rec[k] = dict(eigenli=a.mean(), weighted=b.mean(), p=p, wins=int((b > a).sum()), losses=int((b < a).sum()))
    g = (rec[32]["weighted"] - rec[32]["eigenli"]) / (r["exact"]["uniform"] - rec[32]["eigenli"])
    if rec[32]["eigenli"] >= r["exact"]["uniform"]:
        closed = "n/a"  # EigenLI already at or above full MaxSim
    elif rec[32]["weighted"] >= r["exact"]["uniform"]:
        closed = "above MaxSim"
    else:
        closed = f"{100 * g:.0f}\\%"
    row = f"{MS[m]} & {DS[d]} & {ex:.1f} & " + " & ".join(cells) + f" & {closed} \\\\"
    if d == "arguana":
        out[f"{m}/{d}"] = dict(exact=r["exact"], **{f"k{k}": v for k, v in rec.items()})
        open(f"{TAB}/w1_arguana.tex", "w").write(row + "\n")
        continue
    rows.append(row)
    out[f"{m}/{d}"] = dict(exact=r["exact"], gap_closed_k32=g, **{f"k{k}": v for k, v in rec.items()},
                           idf_plus_shift_k32=R[("idf", True, 32)]["ndcg10"], shift_k32=R[("uniform", True, 32)]["ndcg10"])
open(f"{TAB}/w1.tex", "w").write("\n".join(rows) + "\n")
json.dump(out, open(f"{ROOT}/results/weighted_numbers.json", "w"), indent=1, default=float)
print(open(f"{TAB}/w1.tex").read())
print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk.startswith("k")}
                  for k, v in out.items()}, default=str, indent=0)[:1500])
