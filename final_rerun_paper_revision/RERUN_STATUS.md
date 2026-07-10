# Rerun status — IJIES revision (paper revision experiments)

**Goal:** rerun Stage-1 k-shot inference for M1, M3, M4 on the CORRECTED images (`data2`), 10 seeds, prompt v2.
M2 (ChestXReasoner) skipped — gated model access not yet granted.

## How to check progress
```
cd ~/arta/thoraxai_research/1_equity/final_rerun_paper_revision
python status.py
```
Or tail a log: `tail -f logs/M4.log`

## What is running
- `run_stage1.py M1|M3|M4` — faithful port of experiment-1-v2-data2/M1.ipynb, parametrized.
  - Gemma (M4) loaded in **bfloat16** (fp16 -> all-zero bug fixed).
  - Output: `results/exp1v2_<short>/{checkpoint.jsonl,all_results.jsonl}`. Checkpoint-based, restartable.
- Target per model: 23,650 inferences (5 k-shots x 10 seeds x 473 images).
- ~5-6 s/image on RTX 5090; M3+M4 run in parallel (~82% GPU).

## Status (at launch)
- M4 (medgemma): RUNNING
- M3 (qwen25vl_base): RUNNING
- M1 (nv_reason): FINAL = existing data2 result reused (23,650 records, correct). Decision 2026-06-27: no fresh rerun.

## To restart a model if it dies
```
cd final_rerun_paper_revision
nohup python -u run_stage1.py M4 > logs/M4.log 2>&1 &   # resumes from checkpoint
```
