"""Compact tables and two clean figures for the final report. Every number is read from results/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.environ.get("CERTLI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG, TAB = f"{ROOT}/figs", f"{ROOT}/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
MS = {"colbertv2": "ColBERTv2", "answerai-small": "AnswerAI-small"}
DS = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}
PUB = {("colbertv2", "scifact"): "69.3", ("colbertv2", "nfcorpus"): "33.8",
       ("answerai-small", "scifact"): "74.8", ("answerai-small", "nfcorpus"): "37.3"}
PAIRS = [(m, d) for m in ("colbertv2", "answerai-small") for d in ("scifact", "nfcorpus")]
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
OUT = {}

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300, "text.color": INK})


def signed(x):
    t = f"{x:+.1f}"
    return "0.0" if t in ("+0.0", "-0.0") else t.replace("-", "$-$")


def W(name, rows):
    open(f"{TAB}/{name}.tex", "w").write("\n".join(rows) + "\n")


def cert_rows(m, d):
    rows = []
    for suite in ("main", "bonly", "rq2"):
        r = J(f"{ROOT}/results/cert/{m}__{d}__w1__{suite}.json")
        if r:
            rows += r["rows"]
    return rows


def find(rows, kind, **kw):
    for r in rows:
        if r["kind"] != kind:
            continue
        p = r["rel"]
        if all(p.get(k, {"shift": True, "local_radii": True}.get(k)) == v for k, v in kw.items()):
            return r
    return None


stop = J(f"{ROOT}/results/stop/all.json") or {}
orc = J(f"{ROOT}/results/oracle/all.json") or {}
real = J(f"{ROOT}/results/realized/all.json") or {}


def stop_key(m, d, kind="lrs", rel_has=('"tau": 0.5', '"eps": 0.45')):
    for k, v in stop.items():
        if k.startswith(f"{m}__{d}__w1__{kind}__") and all(x in v["rel"] for x in rel_has):
            return k
    return None


# ------------------------------------------------------------------ Table 1: sanity
rows = []
for m, d in PAIRS:
    r = J(f"{ROOT}/results/exact/{m}__{d}.json")
    rows.append(f"{MS[m]} & {DS[d]} & {100*r['w1']['ndcg@10']:.1f} ({PUB[(m,d)]}) & "
                f"{100*r['idf']['ndcg@10']:.1f} ({signed(100*(r['idf']['ndcg@10']-r['w1']['ndcg@10']))}) & "
                f"${r['lemma1']['max_abs_dev']*1e6:.1f}\\!\\times\\!10^{{-6}}$ & {100*r['lemma1']['frac_queries_identical_top100']:.0f}\\% & "
                f"{100*r['perdoc_centering_ndcg@10']:.1f} \\\\")
    OUT[f"exact/{m}/{d}"] = r["w1"] | {"idf": r["idf"]["ndcg@10"], "lemma1": r["lemma1"], "center": r["perdoc_centering_ndcg@10"]}
W("k1", rows)

# ------------------------------------------------------------------ Table 2: structure (both corpora)
rows = []
for m, d in PAIRS:
    r = J(f"{ROOT}/results/structure/{m}__{d}.json")
    e = r["doc_energy"]
    rows.append(f"{MS[m]} & {DS[d]} & {r['mu_norm']:.2f} & {r['mean_cos_between_docs']:.2f} & "
                f"{100*e['raw']['16']:.0f} / {100*e['shift']['16']:.0f} & {r['doc_eff_rank']['raw']:.1f} / {r['doc_eff_rank']['shift']:.1f} & "
                f"{r['top1_eigvec_abs_cos_with_mu']:.2f} & {r['spearman_idf_resid']:.2f} & "
                f"{r['mean_resid_high_idf_decile']/r['mean_resid_low_idf_decile']:.1f}$\\times$ \\\\")
    OUT[f"structure/{m}/{d}"] = r
W("k2", rows)

# ------------------------------------------------------------------ Table 3: deterministic certification (compact)
rows = []
for m, d in PAIRS:
    cr = cert_rows(m, d)
    cells = []
    for kind, kw in (("lrs", dict(tau=0.3, eps=0.25)), ("lrs", dict(tau=0.5, eps=0.25)), ("lrs", dict(tau=0.5, eps=0.45)),
                     ("lrs", dict(tau=0.5, eps=0.45, local_radii=False)), ("pool", dict(eps=0.45))):
        r = find(cr, kind, **kw)
        cells.append(f"{100*r['frac_read_mean']:.0f} ({100*r['bytes_ratio']:.0f})" if r else "--")
    viol = sum(r["violations_lo"] + r["violations_hi"] for r in cr)
    pairs = sum(r["pairs"] for r in cr)
    rows.append(f"{MS[m]} & {DS[d]} & " + " & ".join(cells) + f" & {viol} / {pairs/1e6:.1f}M \\\\")
    OUT[f"cert/{m}/{d}"] = dict(rows=cr, viol=viol, pairs=pairs)
W("k3", rows)

# ------------------------------------------------------------------ Table 4: why
rows = []
for m, d in PAIRS:
    s = J(f"{ROOT}/results/slack/{m}__{d}.json")
    k = stop_key(m, d)
    rr = real.get(k) if k else None
    rows.append(f"{MS[m]} & {DS[d]} & {s['rms_cos_perp']:.3f} ({s['iso_prediction_perp']:.3f}) & "
                f"{100*s['dev_over_clusterbound_median']:.0f}\\% / {100*s['dev_over_clusterbound_p99']:.0f}\\% & "
                + (f"{rr['gap_k_median']:.3f} & {rr['upper_slack_mean']:.1f} & {100*rr['r_all_p99']:.0f}\\% / {100*rr['r_all_max']:.0f}\\%"
                   if rr else "-- & -- & --") + " \\\\")
    OUT[f"why/{m}/{d}"] = dict(slack=s, realized=rr)
W("k4", rows)

# ------------------------------------------------------------------ Table 5: calibrated certification
rows = []
fig1 = []
for m, d in PAIRS:
    for kind, rel_has, cfg in (("lrs", ('"tau": 0.5', '"eps": 0.45'), dict(tau=0.5, eps=0.45)), ("rq", ('"nbits": 2',), dict(nbits=2))):
        k = stop_key(m, d, kind, rel_has)
        if not k:
            continue
        s = stop[k]; N = s["N"]
        cr = find(cert_rows(m, d), kind, **cfg)
        if cr is None:
            continue
        o = orc.get(k)
        sh, dp = s["shrink"]["ltt05"], s["depth"]["ltt05"]
        lam = s["shrink"]["grid"][int(sh["param_index_median"])]
        code = "LRS" if kind == "lrs" else "RQ 2-bit"
        rows.append(f"{MS[m]} & {DS[d]} & {code} ({100*cr['bytes_ratio']:.0f}\\%) & {100*cr['frac_read_mean']:.1f} & "
                    f"{100*sh['reads']/N:.1f} ({100*sh['miss']:.1f}) & {lam:g} & {100*dp['reads']/N:.1f} ({100*dp['miss']:.1f}) & "
                    + (f"{100*o['oracle_mean']/N:.1f}" if o else "--") + " \\\\")
        OUT[f"ltt/{m}/{d}/{kind}"] = dict(N=N, bytes=cr["bytes_ratio"], exact=cr["frac_read_mean"], shrink=sh, depth=dp,
                                         offset=s["offset"]["ltt05"], shrink10=s["shrink"]["ltt10"], depth10=s["depth"]["ltt10"],
                                         lam=lam, oracle=({kk: o[kk] for kk in o if kk not in ("depth", "dens")} if o else None))
        if kind == "lrs":
            fig1.append(dict(label=f"{MS[m]} / {DS[d]}", size=100 * cr["bytes_ratio"], exact=100 * cr["frac_read_mean"],
                             cal=100 * sh["reads"] / N, cal_miss=100 * sh["miss"], depth=100 * dp["reads"] / N,
                             depth_miss=100 * dp["miss"]))
W("k5", rows)

# ------------------------------------------------------------------ Table 6: EigenLI + shift
rows = []
for m in ("colbertv2", "answerai-small"):
    r = J(f"{ROOT}/results/approx/{m}__scifact.json")
    if not r:
        continue
    ex = r["rows"][0]
    byk = {}
    for x in r["rows"][1:]:
        byk.setdefault(x["k"], {})[x["method"]] = x
    for k in sorted(byk):
        g = byk[k]
        def c(name):
            x = g.get(name)
            return f"{100*x['ndcg10']:.1f} ({100*x['overlap10']:.0f})" if x else "--"
        rows.append(f"{MS[m]} ({100*ex['ndcg10']:.1f}) & {k} & {c('EigenLI')} & {c('EigenLI + shift')} & "
                    f"{c('EigenLI + shift (doc & query)')} & {c('Ward pooling')} \\\\")
    OUT[f"eigen/{m}"] = r["rows"]
W("k6", rows)

json.dump(OUT, open(f"{ROOT}/results/final_numbers.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ Figure 1: reads to return the exact top-10
if fig1:
    n = len(fig1)
    fig, ax = plt.subplots(figsize=(7.0, 0.62 * n + 0.75))
    y = np.arange(n)[::-1] * 1.0
    h = 0.24
    for i, f in enumerate(fig1):
        yy = y[i]
        ax.barh(yy + h, f["exact"], height=h * 0.92, color=MUTED, zorder=3)
        ax.barh(yy, f["depth"], height=h * 0.92, color=ORANGE, zorder=3)
        ax.barh(yy - h, f["cal"], height=h * 0.92, color=BLUE, zorder=3)
        ax.text(f["exact"] * 1.12, yy + h, f"{f['exact']:.0f}%", va="center", fontsize=7.2, color=INK2)
        ax.text(f["depth"] * 1.12, yy, f"{f['depth']:.1f}%   ({f['depth_miss']:.1f}% of queries wrong)", va="center",
                fontsize=7.2, color=INK2)
        ax.text(f["cal"] * 1.12, yy - h, f"{f['cal']:.1f}%   ({f['cal_miss']:.1f}% of queries wrong)", va="center",
                fontsize=7.2, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{f['label']}\ncode = {f['size']:.0f}% of fp16" for f in fig1], fontsize=7.6)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 1200)
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1%", "10%", "100%"])
    ax.spines["bottom"].set_bounds(0.3, 100)
    ax.set_xlabel("share of the corpus read at full precision (log scale)", x=0.36)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=MUTED, label="deterministic certificate (zero risk)"),
                        Patch(color=ORANGE, label="fixed rerank depth, calibrated (risk ≤ 5%)"),
                        Patch(color=BLUE, label="CertLI, calibrated intervals (risk ≤ 5%)")],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7, handlelength=1.0, columnspacing=1.4)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
    fig.savefig(f"{FIG}/f1_reads.pdf"); fig.savefig(f"{FIG}/f1_reads.png", dpi=200)
    plt.close(fig)

# ------------------------------------------------------------------ Figure 2: where the exact score sits inside its interval
hist = J(f"{ROOT}/results/realized/hist.json") or {}
hk = [k for k in hist if k.startswith("colbertv2__scifact")] + [k for k in hist if k.startswith("answerai-small__scifact")]
if hk:
    fig, axes = plt.subplots(1, len(hk), figsize=(3.45 * len(hk), 1.9), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, hk):
        hh = hist[k]
        e = np.array(hh["edges"]); a = np.array(hh["all"], float)
        ctr = 0.5 * (e[1:] + e[:-1]); dens = a / a.sum() * 100
        ax.bar(ctr, dens, width=np.diff(e) * 0.9, color=BLUE, zorder=3)
        if hh.get("lam") is not None:
            ax.axvline(hh["lam"], color=ORANGE, linewidth=1.4, zorder=4)
            ax.text(hh["lam"] + 0.03, dens.max() * 0.82, f"calibrated cut-off\nλ = {hh['lam']:.2f}", color=INK, fontsize=7)
        ax.axvline(1.0, color=INK2, linewidth=1.0, zorder=4)
        ax.text(0.97, dens.max() * 0.82, "deterministic\nbound (λ = 1)", color=INK, fontsize=7, ha="right")
        ax.set_xlim(-0.1, 1.05)
        ax.set_title(hh["title"], fontsize=8, color=INK)
        ax.set_xlabel("position of exact score inside its interval\n(0 = code's estimate, 1 = certified upper bound)")
        ax.set_ylabel("% of (query, doc) pairs")
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    fig.tight_layout(pad=0.3, w_pad=1.5)
    fig.savefig(f"{FIG}/f2_position.pdf"); fig.savefig(f"{FIG}/f2_position.png", dpi=200)
    plt.close(fig)
print("done", sorted(os.listdir(TAB)), sorted(os.listdir(FIG)))
