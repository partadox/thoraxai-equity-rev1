#!/usr/bin/env python3
"""Stage-3 AFTER-inference ensemble (M1+M4) on corrected data2.
Reuses the original ensemble strategies (S1-S5) but with the correct parser."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path("/data/arta/thoraxai_research/1_equity"); RERUN=ROOT/"final_rerun_paper_revision"
sys.path.insert(0, str(ROOT/"src"))     # evaluate.compute_classification_metrics
sys.path.insert(0, str(RERUN))          # parse_dx, ensemble_lib
import ensemble_lib as E
from parse_dx import parse_diagnosis
E.extract_diagnosis = parse_diagnosis   # <-- swap buggy parser for correct one (affects M4)

M1=RERUN/"results/exp1v2_nv_reason/all_results.jsonl"
M4=RERUN/"results/exp1v2_medgemma/all_results.jsonl"
OUT=RERUN/"analysis"; (OUT/"tables").mkdir(parents=True, exist_ok=True)

df=E.load_paired(M1, M4)   # columns pred_m1/pred_m2(=M4), conf_*
pooled=E.compare_all(df, per_seed=False)
perseed=E.compare_all(df, per_seed=True)
pooled.to_csv(OUT/"tables/ensemble_after_pooled.csv", index=False)
perseed.to_csv(OUT/"tables/ensemble_after_per_seed.csv", index=False)

# mean+/-std across seeds per (strategy,test_set,k)
agg=perseed.groupby(["strategy","test_set","k_shot"]).agg(
    f1_mean=("f1","mean"),f1_std=("f1","std"),
    sens_mean=("sensitivity","mean"),spec_mean=("specificity","mean"),
    pbr_mean=("pbr","mean")).reset_index()
agg.to_csv(OUT/"tables/ensemble_after_summary.csv", index=False)

LAB={"M1_single":"M1 Single","M2_single":"M4 Single","S1_unanimous":"S1 AND",
     "S2_any":"S2 OR","S3_detect_verify":"S3 Detect-Review","S4_pbr_weighted":"S4 PBR-Wt","S5_conf_weighted":"S5 Conf-Wt"}
order=["M1_single","M2_single","S1_unanimous","S2_any","S3_detect_verify","S4_pbr_weighted","S5_conf_weighted"]
TS_I="CheXpert Plus (Internal)"; TS_E="RS UNAIR (External)"

def best_k(strat, ts):
    s=agg[(agg.strategy==strat)&(agg.test_set==ts)]
    return s.loc[s.f1_mean.idxmax()] if len(s) else None

print("\n===== TABLE 5 — AFTER-INFERENCE (best-k per strategy) =====")
print(f"{'Strategy':18s} {'bestk':>5} {'F1_int':>7} {'Sens_i':>7} {'Spec_i':>7} {'F1_ext':>7}")
for strat in order:
    bi=best_k(strat,TS_I); be=best_k(strat,TS_E)
    if bi is None: continue
    print(f"{LAB[strat]:18s} {int(bi.k_shot):5d} {bi.f1_mean:7.3f} {bi.sens_mean:7.3f} {bi.spec_mean:7.3f} {be.f1_mean:7.3f}")

print("\n===== OR-gate MONOTONICITY CHECK (same k=8, Internal) — answers Reviewer 1 =====")
k8=agg[(agg.k_shot==8)&(agg.test_set==TS_I)].set_index("strategy")
for strat in ["M1_single","M2_single","S2_any"]:
    r=k8.loc[strat]
    print(f"  {LAB[strat]:12s} Sens={r.sens_mean:.3f} Spec={r.spec_mean:.3f} F1={r.f1_mean:.3f}")
m1=k8.loc["M1_single"]; m4=k8.loc["M2_single"]; s2=k8.loc["S2_any"]
print(f"  -> OR sens {s2.sens_mean:.3f} >= max(M1,M4)={max(m1.sens_mean,m4.sens_mean):.3f}? {s2.sens_mean>=max(m1.sens_mean,m4.sens_mean)-1e-9}")
print(f"  -> OR spec {s2.spec_mean:.3f} <= min(M1,M4)={min(m1.spec_mean,m4.spec_mean):.3f}? {s2.spec_mean<=min(m1.spec_mean,m4.spec_mean)+1e-9}")
print("\nsaved: ensemble_after_{pooled,per_seed,summary}.csv")
