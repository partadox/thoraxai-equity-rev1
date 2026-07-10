#!/usr/bin/env python3
"""Print rerun progress. Run: python status.py"""
import os, glob, subprocess
RERUN=os.path.dirname(os.path.abspath(__file__))
TARGET=23650  # 5 k-shots x 10 seeds x 473 images
MODELS={"M1":"nv_reason","M3":"qwen25vl_base","M4":"medgemma"}
print("="*56); print("  RERUN PROGRESS — final_rerun_paper_revision"); print("="*56)
for mid,short in MODELS.items():
    d=f"{RERUN}/results/exp1v2_{short}"
    final=f"{d}/all_results.jsonl"; ck=f"{d}/checkpoint.jsonl"
    if os.path.exists(final):
        n=sum(1 for _ in open(final)); print(f"  [{mid}] {short:14s} DONE       {n:>6}/{TARGET}")
    elif os.path.exists(ck):
        n=sum(1 for _ in open(ck)); print(f"  [{mid}] {short:14s} running    {n:>6}/{TARGET}  ({100*n/TARGET:4.1f}%)")
    else:
        print(f"  [{mid}] {short:14s} not started")
try:
    g=subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
    print(f"\n  GPU: {g}")
except Exception: pass
