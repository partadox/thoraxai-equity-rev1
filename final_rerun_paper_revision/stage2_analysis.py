#!/usr/bin/env python3
"""
Stage-2 analysis for the corrected (data2) rerun.
Per-model k-shot metrics (mean±std across seeds) + bias (PBR/SDR) + figures.
Uses the format-aware parser parse_dx.parse_diagnosis (NOT src.evaluate).

Outputs under final_rerun_paper_revision/analysis/:
  tables/per_seed_metrics.csv     -- raw per (model,test_set,k,seed): for CI/bootstrap
  tables/summary_kshot.csv        -- mean±std per (model,test_set,k)  [paper Table 3]
  tables/bias_pbr_sdr.csv         -- PBR/SDR per (model,test_set,k)    [paper Table 4]
  figures/<model>_kshot.png       -- F1/Sens/Spec vs k (internal+external)
  figures/f1_compare.png          -- F1 vs k, all models
"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/data/arta/thoraxai_research/1_equity")
RERUN = ROOT / "final_rerun_paper_revision"
sys.path.insert(0, str(RERUN))
from parse_dx import parse_diagnosis

OUT = RERUN / "analysis"; (OUT/"tables").mkdir(parents=True, exist_ok=True); (OUT/"figures").mkdir(parents=True, exist_ok=True)

MODELS = [("M1","nv_reason","NV-Reason-CXR-3B"),
          ("M3","qwen25vl_base","Qwen2.5-VL-3B-Instruct"),
          ("M4","medgemma","MedGemma-4B-IT")]
TEST_SETS = ["CheXpert Plus (Internal)", "RS UNAIR (External)"]
KS = [0,1,2,4,8]

def metrics(gt, pred):
    gt=np.array(gt); pred=np.array(pred)
    tp=int(((pred==1)&(gt==1)).sum()); tn=int(((pred==0)&(gt==0)).sum())
    fp=int(((pred==1)&(gt==0)).sum()); fn=int(((pred==0)&(gt==1)).sum())
    sens=tp/(tp+fn) if tp+fn else 0.0
    spec=tn/(tn+fp) if tn+fp else 0.0
    prec=tp/(tp+fp) if tp+fp else 0.0
    acc=(tp+tn)/len(gt) if len(gt) else 0.0
    f1=2*prec*sens/(prec+sens) if prec+sens else 0.0
    pbr=(tp+fp)/(tp+fn) if tp+fn else 0.0   # predicted pos / actual pos
    return dict(tp=tp,tn=tn,fp=fp,fn=fn,acc=acc,sens=sens,spec=spec,prec=prec,f1=f1,pbr=pbr)

# ---- per-seed metrics ----
rows=[]
for mid,short,name in MODELS:
    f=RERUN/"results"/f"exp1v2_{short}"/"all_results.jsonl"
    df=pd.DataFrame(json.loads(l) for l in open(f))
    df["pred"]=df["generated_text"].apply(parse_diagnosis)
    for ts in TEST_SETS:
        for k in KS:
            for seed,g in df[(df.test_set==ts)&(df.k_shot==k)].groupby("seed"):
                m=metrics(g.ground_truth.tolist(), g.pred.tolist())
                rows.append(dict(model=mid,model_name=name,test_set=ts,k_shot=k,seed=int(seed),**m))
per_seed=pd.DataFrame(rows)
per_seed.to_csv(OUT/"tables"/"per_seed_metrics.csv", index=False)

# ---- summary mean±std ----
agg=[]
for (mid,ts,k),g in per_seed.groupby(["model","test_set","k_shot"]):
    d=dict(model=mid,test_set=ts,k_shot=k,n_seeds=len(g))
    for col in ["acc","sens","spec","prec","f1","pbr"]:
        d[f"{col}_mean"]=g[col].mean(); d[f"{col}_std"]=g[col].std()
    agg.append(d)
summary=pd.DataFrame(agg).sort_values(["model","test_set","k_shot"])
summary.to_csv(OUT/"tables"/"summary_kshot.csv", index=False)

# ---- bias: PBR + SDR (SDR per seed then mean) ----
bias=[]
for (mid,ts),g in per_seed.groupby(["model","test_set"]):
    spec0={s:v for s,v in g[g.k_shot==0].set_index("seed")["spec"].items()}
    for k in KS:
        gk=g[g.k_shot==k]
        sdr=[]
        for _,r in gk.iterrows():
            s0=spec0.get(r["seed"],None)
            sdr.append(((s0-r["spec"])/s0*100) if s0 and s0>0 else 0.0)
        bias.append(dict(model=mid,test_set=ts,k_shot=k,
                         pbr_mean=gk.pbr.mean(),pbr_std=gk.pbr.std(),
                         spec_mean=gk.spec.mean(),
                         sdr_mean=float(np.mean(sdr)),sdr_std=float(np.std(sdr)),
                         sens_mean=gk.sens.mean()))
bias=pd.DataFrame(bias).sort_values(["model","test_set","k_shot"])
bias.to_csv(OUT/"tables"/"bias_pbr_sdr.csv", index=False)

# ---- figures ----
COL={"CheXpert Plus (Internal)":"#1f77b4","RS UNAIR (External)":"#ff7f0e"}
for mid,short,name in MODELS:
    fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    for ax,met,title in zip(axes,["f1","sens","spec"],["F1","Sensitivity","Specificity"]):
        for ts in TEST_SETS:
            s=summary[(summary.model==mid)&(summary.test_set==ts)].sort_values("k_shot")
            ax.errorbar(s.k_shot,s[f"{met}_mean"],yerr=s[f"{met}_std"],marker="o",capsize=3,
                        color=COL[ts],label="Internal" if "Internal" in ts else "External")
        ax.set_xlabel("k-shot"); ax.set_title(title); ax.set_ylim(-0.02,1.02); ax.set_xticks(KS); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{mid} — {name}",fontsize=11); fig.tight_layout()
    fig.savefig(OUT/"figures"/f"{mid}_kshot.png",dpi=150,bbox_inches="tight")
    fig.savefig(OUT/"figures"/f"{mid}_kshot.pdf",bbox_inches="tight"); plt.close(fig)

# combined F1 comparison
fig,axes=plt.subplots(1,2,figsize=(10,3.6))
for ax,ts in zip(axes,TEST_SETS):
    for mid,short,name in MODELS:
        s=summary[(summary.model==mid)&(summary.test_set==ts)].sort_values("k_shot")
        ax.errorbar(s.k_shot,s.f1_mean,yerr=s.f1_std,marker="o",capsize=3,label=mid)
    ax.set_title(ts);ax.set_xlabel("k-shot");ax.set_ylabel("F1");ax.set_ylim(-0.02,1.02);ax.set_xticks(KS);ax.grid(alpha=.3);ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/"figures"/"f1_compare.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---- console: paper-style Table 3 (F1) + Table 4 (bias M1,M4) ----
def cell(r): return f"{r.f1_mean:.3f}±{r.f1_std:.3f}"
print("\n===== TABLE 3 — F1 (mean±std) =====")
for ts in TEST_SETS:
    print(f"\n{ts}")
    for mid,short,name in MODELS:
        s=summary[(summary.model==mid)&(summary.test_set==ts)].sort_values("k_shot")
        print(f"  {mid}: "+"  ".join(f"k{int(r.k_shot)}={cell(r)}" for _,r in s.iterrows()))
print("\n===== TABLE 4 — PBR / SDR (M1, M4, Internal) =====")
for mid in ["M1","M4"]:
    b=bias[(bias.model==mid)&(bias.test_set=="CheXpert Plus (Internal)")].sort_values("k_shot")
    print(f"  {mid}: "+"  ".join(f"k{int(r.k_shot)}[PBR={r.pbr_mean:.2f},SDR={r.sdr_mean:.0f}%]" for _,r in b.iterrows()))
print("\nSaved tables -> analysis/tables/ , figures -> analysis/figures/")
print("files:", [p.name for p in sorted((OUT/'tables').iterdir())], [p.name for p in sorted((OUT/'figures').glob('*.png'))])
