# ThoraxAI-Equity — Corrected Rerun Code

Experiment code for the corrected re-run of the paper:

**"Comprehensive Evaluation of Few-Shot In-Context Learning and Dual-Agent Ensemble Strategies for Pneumothorax Detection Using Resource-Efficient Vision-Language Models"**

## Overview

This folder contains the code used to reproduce, from a single consistent image set, the individual in-context-learning (ICL) evaluation, bias profiling, and dual-agent ensemble experiments for four resource-efficient vision-language models (VLMs, 3–4B parameters, 4-bit NF4 quantized). It also includes the corrected output parser and the per-run result tables (CSV). Large raw prediction files (`all_results.jsonl`), model weights, and images are not included here.

## Models

All models are 3–4B parameters and quantized to 4-bit NF4, so the full pipeline runs on a single consumer GPU.

| ID | Model | Parameters | Training Strategy |
|----|-------|-----------|-------------------|
| M1 | [NV-Reason-CXR-3B](https://github.com/NVIDIA-Medtech/NV-Reason-CXR) | 3B | CXR-reasoning fine-tune |
| M2 | [ChestXReasoner](https://github.com/MAGIC-AI4Med/ChestX-Reasoner) | 3B | CXR-direct fine-tune (gated; not re-run) |
| M3 | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3B | No medical fine-tune |
| M4 | [MedGemma-4B-IT](https://huggingface.co/google/medgemma-4b-it) | 4B | Medical-general fine-tune (requires bfloat16 compute) |

M2 is a gated model; it was not re-run on the corrected images. It is a degenerate all-positive predictor whose F1 is fixed by class prevalence and is therefore image-independent.

## Pipeline structure

```
├── configs/
│   └── experiment_config2.yaml    # k = {0,1,2,4,8}, 10 seeds, generation params
│
├── src/                           # Core source modules
│   ├── inference.py               # MultiModelEngine (qwen / gemma backends)
│   ├── prompt_builder.py          # Prompt construction (system, user, few-shot)
│   ├── example_selector.py        # Few-shot example sampling
│   ├── evaluate.py                # Metrics computation
│   └── run_experiments.py         # Original single-model runner
│
└── final_rerun_paper_revision/
    ├── run_stage1.py              # Stage 1: individual ICL inference, M1/M3/M4
    │                              #   usage: python run_stage1.py M1|M3|M4 [--sanity]
    │                              #   Gemma loaded in bfloat16; checkpoint-based / restartable
    ├── parse_dx.py                # Format-aware verdict parser (see Notes)
    ├── stage2_analysis.py         # Stage 2: per-seed metrics, PBR/SDR/BCS, figures
    ├── ensemble_lib.py            # Ensemble strategies S1–S5 (parser set to parse_dx)
    ├── stage3_after.py            # Stage 3A: after-inference ensemble (Table 5)
    ├── during_ensemble.py         # Reviewer-prompt helpers for during-inference
    ├── stage3_during.py           # Stage 3C: M1 -> M4 sequential review (prompt v3)
    ├── stage3_during_eval.py      # Stage 3C: D1–D4 aggregation + confidence calibration (Table 6)
    ├── stage4_stats.py            # 95% CIs, paired Wilcoxon, decision-curve analysis
    ├── stage_figures.py           # Regenerates the data figures
    ├── status.py                  # Progress monitor
    ├── CORRECTED_RESULTS.md        # Consolidated corrected results
    └── analysis/tables/*.csv       # Result tables (Tables 3–6 + per-seed metrics)
```

## Datasets

Datasets are available on Kaggle: **[Pneumothorax Chest X-ray Indonesia](https://www.kaggle.com/datasets/artakusuma/pneumothorax-chest-x-ray-indonesia)**

- **[CheXpert Plus](https://aimi.stanford.edu/datasets/chexpert-plus)** (Stanford) — Few-shot pool (300 images) + Internal test (200 images)
- **RS UNAIR** (Universitas Airlangga Hospital, Surabaya) — External test (278 images)

## Setup

```bash
pip install -r requirements.txt
```

Set `PROJECT_ROOT` at the top of each script to the directory that contains `configs/`, `src/`, `data/`, and `models/` on your machine.

## Hardware

The pipeline is designed to run on a **single consumer GPU**. Because all models are 4-bit quantized (≈2–3 GB each), they fit within **8 GB of VRAM**; a higher-memory GPU only reduces wall-clock time.

## Key Results (corrected image set)

| Approach | Strategy | F1 (Internal) | F1 (External) |
|----------|----------|---------------|---------------|
| Single | M1 best (k=8) | 0.660 | 0.812 |
| Single | M4 best (k=8) | 0.700 | **0.921** |
| After-Inference | S2 OR-gate | **0.720** | 0.898 |
| During-Inference | D4 OR-gate | 0.691 | 0.876 |

At a matched operating point (k=8), the OR-gate ensemble is statistically indistinguishable from the single reviewer M4 (paired Wilcoxon p = 1.00); it significantly outperforms only the weaker detector M1. After-inference voting consistently outperforms during-inference sequential review at lower cost.

## Notes

- **Parser (`parse_dx.py`).** The evaluation parser is format-aware: for M1 it reads the finding list inside the `<answer>` tag, and for the prose-style models (M3, M4) it reads the CLASSIFICATION verdict with negation handling, so that "NO PNEUMOTHORAX DETECTED" is correctly counted as negative.
- **Reproducibility.** Stage-1 inference is checkpoint-based and restartable. Result tables under `analysis/tables/` correspond directly to the tables in the manuscript.

## Citation

```
[To be added after publication]
```

## Acknowledgments

This work was supported by the Indonesia Endowment Fund for Education (LPDP) through the EQUITY-WCU program (ITS-LPDP-Kemendiktisaintek).
