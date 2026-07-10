#!/usr/bin/env python3
"""Bootstrap CIs + paired tests + decision-curve analysis (R2.2, R2.7)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.size":12,"font.family":"serif","axes.grid":True,
    "grid.alpha":0.3,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
ROOT=Path("/data/arta/thoraxai_research/1_equity"); RERUN=ROOT/"final_rerun_paper_revision"
T=RERUN/"analysis/tables"; OUT=RERUN/"analysis/figures_paper/conf"
TS_I="CheXpert Plus (Internal)"; TS_E="RS UNAIR (External)"

def ci95(x):
    x=np.asarray(x,float); n=len(x); m=x.mean()
    if n<2: return m,m,m
    h=stats.t.ppf(0.975,n-1)*x.std(ddof=1)/np.sqrt(n)
    return m, m-h, m+h

ps=pd.read_csv(T/"ensemble_after_per_seed.csv")  # strategy,k_shot,test_set,seed,f1,...
# best internal k per strategy (by mean f1)
def bestk(strat):
    g=ps[(ps.strategy==strat)&(ps.test_set==TS_I)].groupby("k_shot").f1.mean()
    return int(g.idxmax())

print("===== 95% CI of F1 (per-seed, t-based) =====")
LAB={"M1_single":"M1","M2_single":"M4","S2_any":"S2-OR","S1_unanimous":"S1"}
for strat in ["M1_single","M2_single","S2_any"]:
    k=bestk(strat)
    for ts,tn in [(TS_I,"Int"),(TS_E,"Ext")]:
        # report at internal-best-k for both (fixed-k, fair)
        v=ps[(ps.strategy==strat)&(ps.test_set==ts)&(ps.k_shot==k)].f1.values
        m,lo,hi=ci95(v)
        print(f"  {LAB[strat]:6s} @k={k} {tn}: F1={m:.3f} [95% CI {lo:.3f}, {hi:.3f}] (n={len(v)})")

print("\n===== Paired difference (fixed k=8, internal) — fair same-k comparison =====")
def perseed(strat,k,ts):
    return ps[(ps.strategy==strat)&(ps.test_set==ts)&(ps.k_shot==k)].sort_values("seed").f1.values
for a,b in [("S2_any","M2_single"),("M2_single","M1_single"),("S2_any","M1_single")]:
    da=perseed(a,8,TS_I); db=perseed(b,8,TS_I); d=da-db
    m,lo,hi=ci95(d)
    try: w=stats.wilcoxon(da,db).pvalue
    except Exception: w=float("nan")
    print(f"  {LAB[a]} - {LAB[b]}: Δ={m:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}], Wilcoxon p={w:.4f}")

# ===== Decision-curve analysis (internal) =====
# Binary test net benefit: NB(pt) = sens*prev - (1-spec)*(1-prev)*(pt/(1-pt))
prev=0.5  # internal 100/200
def nb_curve(sens,spec,pts):
    return sens*prev - (1-spec)*(1-prev)*(pts/(1-pts))
# operating points at k=8 internal (from summary): M4 sens .712 spec .683 ; S2 (k=8) sens .928 spec .273 ; M1 sens .796 spec .390
ops={"M4 (single)":(0.712,0.683),"S2 OR-gate (k=8)":(0.928,0.273),"M1 (single)":(0.796,0.390)}
pts=np.linspace(0.01,0.6,100)
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(pts, prev - (1-prev)*(pts/(1-pts)), "k-", lw=1, label="Treat all")
ax.plot(pts, np.zeros_like(pts), "k:", lw=1, label="Treat none")
for name,(se,sp) in ops.items():
    ax.plot(pts, nb_curve(se,sp,pts), label=name)
ax.set_xlabel("Threshold probability $p_t$"); ax.set_ylabel("Net benefit")
ax.set_ylim(-0.05,0.5); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/"fig_decision_curve.png",bbox_inches="tight"); plt.close(fig)
print("\nsaved decision-curve fig -> analysis/figures_paper/conf/fig_decision_curve.png")

# FP burden at operating points (internal, per 100 patients, prev 50%)
print("\n===== False-positive burden (internal, per 100 patients) =====")
for name,(se,sp) in ops.items():
    fp=(1-sp)*50; tp=se*50; fn=(1-se)*50
    print(f"  {name}: TP={tp:.0f} FN={fn:.0f} FP={fp:.0f}  (of 50 PTX / 50 normal)")
