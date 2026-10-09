"""Tables for the one-page EigenLI comparison (report/eigenli_vs_certli.tex). Every number is read from results/."""
import json
import os

ROOT = os.environ.get("CERTLI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = f"{ROOT}/tables"
MS = {"colbertv2": "ColBERTv2", "answerai-small": "AnswerAI-small"}
DS = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}
PAIRS = [(m, d) for m in ("colbertv2", "answerai-small") for d in ("scifact", "nfcorpus")]
J = lambda p: json.load(open(p)) if os.path.exists(p) else None


def f1(x):
    return f"{100 * x:.1f}"


def delta(x, ref):
    dd = round(100 * x, 1) - round(100 * ref, 1)  # difference of the printed values
    s = f"{dd:+.1f}".replace("-", "$-$") if abs(dd) >= 0.05 else "0.0"
    cell = f"{100 * x:.1f} ({s})"
    return f"\\textbf{{{cell}}}" if dd > 0.5 else cell


def lrs_point_estimate(m, d):
    for suite in ("main", "bonly"):
        r = J(f"{ROOT}/results/cert/{m}__{d}__w1__{suite}.json")
        if not r:
            continue
        for row in r["rows"]:
            p = row["rel"]
            if row["kind"] == "lrs" and p.get("tau") == 0.5 and p.get("eps") == 0.45 and p.get("shift", True) \
                    and p.get("local_radii", True):
                return row
    return None


e1, e2, out = [], [], {}
for m, d in PAIRS:
    er = J(f"{ROOT}/results/eigen_rescore/{m}__{d}.json")
    es = J(f"{ROOT}/results/eigen_sparse/{m}__{d}.json")
    if not er:
        continue
    ex = er["exact_ndcg10"]
    e16, e32, c32 = er["eigenli16"], er["eigenli32"], er.get("eigenli32_centred")
    sp = None
    if es:
        for row in es["rows"]:
            if row.get("budget") == 32 and row.get("preregistered"):
                sp = row
    e1.append(f"{MS[m]} & {DS[d]} & {f1(ex)} & {f1(e16['ndcg10'])} & {f1(e32['ndcg10'])} & "
              f"{delta(c32['ndcg10'], e32['ndcg10']) if c32 else '--'} & {delta(sp['ndcg10'], e32['ndcg10']) if sp else '--'} \\\\")
    lp = lrs_point_estimate(m, d)
    ce = er.get("certli")
    e2.append(f"{MS[m]} & {DS[d]} & {f1(e32['ndcg10'])} & {f1(lp['approx_ndcg10'])} ({100 * lp['bytes_ratio']:.0f}\\%) & "
              f"{f1(ce['rescore']['ndcg10'])} ({100 * ce['rescore']['reads_frac']:.1f}\\%) & "
              f"{f1(e32['rescore']['ndcg10'])} ({100 * e32['rescore']['reads_frac']:.1f}\\%) & {f1(ex)} \\\\")
    out[f"{m}/{d}"] = dict(exact=ex, eigenli16=e16["ndcg10"], eigenli32=e32["ndcg10"],
                           centred32=c32["ndcg10"] if c32 else None, sparse_24_8=sp["ndcg10"] if sp else None,
                           lrs_point=lp["approx_ndcg10"], lrs_memory=lp["bytes_ratio"],
                           certli_cal=ce["rescore"], eigen32_cal=e32["rescore"], eigen16_cal=e16["rescore"],
                           eigen32_memory=e32["memory_frac"])
open(f"{TAB}/e1.tex", "w").write("\n".join(e1) + "\n")
open(f"{TAB}/e2.tex", "w").write("\n".join(e2) + "\n")
json.dump(out, open(f"{ROOT}/results/eigen_numbers.json", "w"), indent=1)
print(open(f"{TAB}/e1.tex").read()); print(open(f"{TAB}/e2.tex").read())
