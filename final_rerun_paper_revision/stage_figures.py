#!/usr/bin/env python3
"""Generate corrected (data2) paper figures matching the draft's filenames.
Output: analysis/figures_paper/{M1,M4,ens,conf}/... (publication style)."""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.size":12,"font.family":"serif","axes.grid":True,
    "grid.alpha":0.3,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
ROOT=Path("/data/arta/thoraxai_research/1_equity"); RERUN=ROOT/"final_rerun_paper_revision"
sys.path.insert(0,str(RERUN)); from parse_dx import parse_diagnosis
T=RERUN/"analysis/tables"; OUT=RERUN/"analysis/figures_paper"
for s in ["M1","M4","ens","conf"]: (OUT/s).mkdir(parents=True,exist_ok=True)
TS_I="CheXpert Plus (Internal)"; TS_E="RS UNAIR (External)"; KS=[0,1,2,4,8]
CI,CE="#1f77b4","#d62728"
sm=pd.read_csv(T/"summary_kshot.csv")

def sub(mid,ts): return sm[(sm.model==mid)&(sm.test_set==ts)].sort_values("k_shot")

# ── Fig 5 (per model): F1 vs k ──
for mid in ["M1","M4"]:
    fig,ax=plt.subplots(figsize=(5.2,3.6))
    for ts,c,lb in [(TS_I,CI,"Internal"),(TS_E,CE,"External")]:
        s=sub(mid,ts); ax.errorbar(s.k_shot,s.f1_mean,yerr=s.f1_std,marker="o",capsize=3,color=c,label=lb)
    ax.set_xlabel("k-shot"); ax.set_ylabel("F1 score"); ax.set_ylim(-0.02,1.02); ax.set_xticks(KS); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/mid/"fig1_kshot_accuracy_f1.png",bbox_inches="tight"); plt.close(fig)

# ── Fig 6 (per model): sensitivity & specificity vs k ──
for mid in ["M1","M4"]:
    fig,ax=plt.subplots(figsize=(5.2,3.6))
    for ts,c,lb in [(TS_I,CI,"Int"),(TS_E,CE,"Ext")]:
        s=sub(mid,ts)
        ax.plot(s.k_shot,s.sens_mean,marker="o",color=c,label=f"Sensitivity ({lb})")
        ax.plot(s.k_shot,s.spec_mean,marker="s",ls="--",color=c,label=f"Specificity ({lb})")
    ax.set_xlabel("k-shot"); ax.set_ylabel("Score"); ax.set_ylim(-0.02,1.02); ax.set_xticks(KS); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/mid/"fig2_sensitivity_specificity.png",bbox_inches="tight"); plt.close(fig)

# ── Fig 10 (per model): generalization gap = internal vs external F1 ──
for mid in ["M1","M4"]:
    fig,ax=plt.subplots(figsize=(5.2,3.6))
    si=sub(mid,TS_I); se=sub(mid,TS_E)
    gap=si.f1_mean.values - se.f1_mean.values
    ax.plot(KS,si.f1_mean,marker="o",color=CI,label="Internal F1")
    ax.plot(KS,se.f1_mean,marker="o",color=CE,label="External F1")
    ax.bar(KS,gap,width=0.4,color="grey",alpha=0.4,label="Gap (Int-Ext)")
    ax.axhline(0,color="k",lw=0.6); ax.set_xlabel("k-shot"); ax.set_ylabel("F1 / gap")
    ax.set_xticks(KS); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/mid/"fig7_generalization_gap.png",bbox_inches="tight"); plt.close(fig)

# ── Fig 7 (ensemble): F1 comparison across strategies (best-k) ──
af=pd.read_csv(T/"ensemble_after_summary.csv")
LAB={"M1_single":"M1","M2_single":"M4","S1_unanimous":"S1","S2_any":"S2-OR",
     "S3_detect_verify":"S3","S4_pbr_weighted":"S4","S5_conf_weighted":"S5"}
order=list(LAB);
def bestf1(strat,ts):
    x=af[(af.strategy==strat)&(af.test_set==ts)]; return x.f1_mean.max()
fig,ax=plt.subplots(figsize=(6.8,3.6))
x=np.arange(len(order)); w=0.38
ax.bar(x-w/2,[bestf1(s,TS_I) for s in order],w,color=CI,label="Internal")
ax.bar(x+w/2,[bestf1(s,TS_E) for s in order],w,color=CE,label="External")
ax.set_xticks(x); ax.set_xticklabels([LAB[s] for s in order]); ax.set_ylabel("Best-k F1"); ax.set_ylim(0,1.0); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"ens"/"fig1_f1_comparison.png",bbox_inches="tight"); plt.close(fig)

# ── Fig 8/9 (confidence) from during_sd_v3 ──
def extract_conf(t):
    for p in [r'\*{0,2}CONFIDENCE\*{0,2}\s*[:=]\s*(\d{1,3})\s*%',r'\*{0,2}CONFIDENCE\*{0,2}\s*[:=]\s*(\d{1,3})(?!\d)']:
        m=re.search(p,t,re.I)
        if m and float(m.group(1))<=100: return float(m.group(1))
    return -1.0
dur=pd.DataFrame(json.loads(l) for l in open(RERUN/"results/during_sd_v3/all_results.jsonl"))
dur["pred"]=dur.m4_generated_text.apply(parse_diagnosis); dur["conf"]=dur.m4_generated_text.apply(extract_conf)
v=dur[dur.conf>=0].copy(); v["correct"]=v.pred==v.ground_truth
# fig4 distribution
fig,ax=plt.subplots(figsize=(5.2,3.6))
bins=np.arange(0,105,5)
ax.hist(v[v.correct].conf,bins=bins,alpha=0.6,color=CI,label=f"Correct (mean {v[v.correct].conf.mean():.1f}%)")
ax.hist(v[~v.correct].conf,bins=bins,alpha=0.6,color=CE,label=f"Incorrect (mean {v[~v.correct].conf.mean():.1f}%)")
ax.set_xlabel("Self-reported confidence (%)"); ax.set_ylabel("Count"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/"conf"/"fig4_confidence_distribution.png",bbox_inches="tight"); plt.close(fig)
# fig5 calibration (bin accuracy vs confidence)
fig,ax=plt.subplots(figsize=(5.2,3.6))
v["bin"]=pd.cut(v.conf,bins=np.arange(0,110,10))
cal=v.groupby("bin",observed=True).agg(acc=("correct","mean"),conf=("conf","mean"),n=("correct","size")).dropna()
ax.plot([0,1],[0,1],"k--",lw=0.8,label="Perfect calibration")
ax.plot(cal.conf/100,cal.acc,marker="o",color=CI,label="M4 reviewer")
ax.set_xlabel("Mean confidence"); ax.set_ylabel("Accuracy"); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/"conf"/"fig5_calibration.png",bbox_inches="tight"); plt.close(fig)

print("generated figures:")
for p in sorted(OUT.rglob("*.png")): print("  ",p.relative_to(OUT))
