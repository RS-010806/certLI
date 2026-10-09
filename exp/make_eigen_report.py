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


def r1(x):
    return round(100 * x, 1)


def cell(v, bold, extra=""):
    t = f"{v:.1f}{extra}"
    return f"\\textbf{{{t}}}" if bold else t


def signed(dd):
    return "0.0" if abs(dd) < 0.05 else f"{dd:+.1f}".replace("-", "$-$")


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
    # Table 1: equal memory (32 vectors per document); bold = best of the three; change vs EigenLI in brackets
    a, b, c = r1(e32["ndcg10"]), r1(c32["ndcg10"]), r1(sp["ndcg10"])
    best = max(a, b, c)
    e1.append(f"{MS[m]} & {DS[d]} & {cell(a, a == best)} & {cell(b, b == best, f' ({signed(b - a)})')} & "
              f"{cell(c, c == best, f' ({signed(c - a)})')} & {r1(ex):.1f} \\\\")
    # Table 2: calibrated exact rescoring, same guarantee; bold = better of the pair
    ce = er.get("certli")
    n_e, n_c = r1(e32["rescore"]["ndcg10"]), r1(ce["rescore"]["ndcg10"])
    f_e, f_c = 100 * e32["rescore"]["reads_frac"], 100 * ce["rescore"]["reads_frac"]
    ratio = round(f_e, 1) / round(f_c, 1)  # ratio of the printed percentages
    e2.append(f"{MS[m]} & {DS[d]} & {cell(n_e, n_e >= n_c)} & {f_e:.1f}\\% & {cell(n_c, n_c >= n_e)} & "
              f"\\textbf{{\\boldmath {f_c:.1f}\\% ({ratio:.1f}$\\times$ fewer)}} & {r1(ex):.1f} \\\\")
    lp = lrs_point_estimate(m, d)
    out[f"{m}/{d}"] = dict(exact=ex, eigenli16=e16["ndcg10"], eigenli32=e32["ndcg10"],
                           centred32=c32["ndcg10"] if c32 else None, sparse_24_8=sp["ndcg10"] if sp else None,
                           lrs_point=lp["approx_ndcg10"], lrs_memory=lp["bytes_ratio"],
                           certli_cal=ce["rescore"], eigen32_cal=e32["rescore"], eigen16_cal=e16["rescore"],
                           eigen32_memory=e32["memory_frac"], rescore_ratio=ratio)
open(f"{TAB}/e1.tex", "w").write("\n".join(e1) + "\n")
open(f"{TAB}/e2.tex", "w").write("\n".join(e2) + "\n")
json.dump(out, open(f"{ROOT}/results/eigen_numbers.json", "w"), indent=1)
print(open(f"{TAB}/e1.tex").read()); print(open(f"{TAB}/e2.tex").read())
