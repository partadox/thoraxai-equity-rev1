#!/usr/bin/env python3
"""D1-D4 aggregation + confidence calibration for during-inference v3 (Table 6/7)."""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path("/data/arta/thoraxai_research/1_equity"); RERUN=ROOT/"final_rerun_paper_revision"
sys.path.insert(0, str(RERUN))
from parse_dx import parse_diagnosis
OUT=RERUN/"analysis"; (OUT/"tables").mkdir(parents=True, exist_ok=True)

def extract_confidence(text):
    if not text or not text.strip(): return -1.0
    for pat in [r'\*{0,2}CONFIDENCE\*{0,2}\s*[:=]\s*(\d{1,3})\s*%',
                r'\*{0,2}CONFIDENCE\*{0,2}\s*[:=]\s*(\d{1,3})(?!\d)',
                r'(\d{1,3})\s*%\s*confiden']:
        m=re.search(pat,text,re.I)
        if m:
            v=float(m.group(1))
            if v<=100: return v
    return -1.0

def metrics(gt,pred):
    gt=np.array(gt);pred=np.array(pred)
    tp=int(((pred==1)&(gt==1)).sum());tn=int(((pred==0)&(gt==0)).sum())
    fp=int(((pred==1)&(gt==0)).sum());fn=int(((pred==0)&(gt==1)).sum())
    se=tp/(tp+fn) if tp+fn else 0; sp=tn/(tn+fp) if tn+fp else 0
    pr=tp/(tp+fp) if tp+fp else 0; f1=2*pr*se/(pr+se) if pr+se else 0
    return dict(sens=se,spec=sp,f1=f1,acc=(tp+tn)/len(gt))

df=pd.DataFrame(json.loads(l) for l in open(RERUN/"results/during_sd_v3/all_results.jsonl"))
df["m4_pred"]=df.m4_generated_text.apply(parse_diagnosis)
df["m1_pred"]=df.m1_generated_text.apply(parse_diagnosis)
df["m4_conf"]=df.m4_generated_text.apply(extract_confidence)

# D-strategies
df["d1"]=df.m4_pred
def d2(r): return r.m4_pred if r.m4_conf<0 else (r.m4_pred if r.m4_conf>=60 else r.m1_pred)
def d3(r):
    if r.m4_conf<0: return r.m4_pred
    if r.m4_conf>=80: return r.m4_pred
    if r.m4_conf<=40: return r.m1_pred
    return max(r.m1_pred,r.m4_pred)
def d4(r): return r.m1_pred if r.m1_pred==r.m4_pred else max(r.m1_pred,r.m4_pred)
df["d2"]=df.apply(d2,axis=1); df["d3"]=df.apply(d3,axis=1); df["d4"]=df.apply(d4,axis=1)

# per-seed metrics -> mean per (strategy,test_set,k)
rows=[]
for strat in ["m1_pred","d1","d2","d3","d4"]:
    for ts,gt_ts in df.groupby("test_set"):
        for k,gk in gt_ts.groupby("k_shot"):
            per=[metrics(g.ground_truth.tolist(),g[strat].tolist()) for _,g in gk.groupby("seed")]
            rows.append(dict(strategy=strat,test_set=ts,k_shot=k,
                             f1=np.mean([m["f1"] for m in per]),
                             sens=np.mean([m["sens"] for m in per]),
                             spec=np.mean([m["spec"] for m in per])))
agg=pd.DataFrame(rows); agg.to_csv(OUT/"tables/ensemble_during_summary.csv",index=False)

LAB={"m1_pred":"M1 Single","d1":"D1 M4-Review","d2":"D2 Conf-Wt","d3":"D3 Threshold","d4":"D4 OR-gate"}
TS_I="CheXpert Plus (Internal)"; TS_E="RS UNAIR (External)"
def bestk(s,ts):
    x=agg[(agg.strategy==s)&(agg.test_set==ts)]; return x.loc[x.f1.idxmax()]
print("\n===== TABLE 6 — DURING-INFERENCE v3 (best-k) =====")
print(f"{'Strategy':14s} {'k':>2} {'F1_int':>7} {'Sens_i':>7} {'Spec_i':>7} {'F1_ext':>7}")
for s in ["m1_pred","d1","d2","d3","d4"]:
    bi=bestk(s,TS_I); be=bestk(s,TS_E)
    print(f"{LAB[s]:14s} {int(bi.k_shot):2d} {bi.f1:7.3f} {bi.sens:7.3f} {bi.spec:7.3f} {be.f1:7.3f}")

# confidence calibration
v=df[df.m4_conf>=0].copy(); v["correct"]=(v.m4_pred==v.ground_truth)
print("\n===== CONFIDENCE CALIBRATION (M4 reviewer) =====")
print(f"  found {len(v)}/{len(df)} | mean={v.m4_conf.mean():.1f}% std={v.m4_conf.std():.1f}%")
print(f"  when CORRECT:   {v[v.correct].m4_conf.mean():.1f}%  (n={v.correct.sum()})")
print(f"  when INCORRECT: {v[~v.correct].m4_conf.mean():.1f}%  (n={(~v.correct).sum()})")
print(f"  -> {'HIGHER when wrong (miscalibrated)' if v[~v.correct].m4_conf.mean()>v[v.correct].m4_conf.mean() else 'higher when correct'}")

# internal contradictions: M4 mentions pneumothorax signs in text but classifies negative
def has_ptx_signs(t):
    t=t.lower()
    return any(w in t for w in ["visceral pleural line","absent lung marking","pneumothorax is present",
                                "consistent with pneumothorax","suggestive of pneumothorax"])
df["contra"]=df.apply(lambda r: has_ptx_signs(r.m4_generated_text) and r.m4_pred==0, axis=1)
print(f"\n  internal contradictions (signs mentioned but classified NEGATIVE): {int(df.contra.sum())}/{len(df)}")
print("\nsaved: ensemble_during_summary.csv")
