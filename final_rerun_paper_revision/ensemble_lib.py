from __future__ import annotations

"""
Dual-Agent Ensemble Aggregation for P1 (Q2 INASS Paper).

Combines M1 (Detector — high sensitivity, over-predicts) with
M2 (Verifier — high specificity, under-predicts) using 5 strategies:

    S1  Unanimous Positive    — PTX only if BOTH models predict PTX
    S2  Any Positive          — PTX if EITHER model predicts PTX
    S3  Detector→Verifier     — M1 screens; if positive, M2 verifies
    S4  PBR-Guided Weighted   — weight inversely proportional to |PBR−1|
    S5  Confidence-Weighted   — weight by answer-extraction certainty

All strategies operate on pre-extracted binary predictions (0/1)
from existing Exp1 v2 results — no new inference required.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import extract_diagnosis, compute_classification_metrics

# ── Constants ─────────────────────────────────────────────────────────────

STRATEGIES = ["S1_unanimous", "S2_any", "S3_detect_verify", "S4_pbr_weighted", "S5_conf_weighted"]

M1_ID = "M1"
M2_ID = "M2"


# ── Data Loading ──────────────────────────────────────────────────────────

def load_results(jsonl_path: str | Path) -> pd.DataFrame:
    """Load a JSONL results file into a DataFrame with extracted predictions."""
    records = [json.loads(line) for line in open(jsonl_path, encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
    df["predicted"] = df["generated_text"].apply(extract_diagnosis)
    df["confidence"] = df["generated_text"].apply(extract_confidence)
    return df


def load_paired(m1_path: str | Path, m2_path: str | Path) -> pd.DataFrame:
    """
    Load M1 and M2 results, align by (k_shot, seed, test_set, image_id),
    and return a single DataFrame with columns for both models.
    """
    df_m1 = load_results(m1_path)
    df_m2 = load_results(m2_path)

    # Keep only alignment keys + model-specific columns
    keys = ["k_shot", "seed", "test_set", "image_id", "ground_truth"]

    m1 = df_m1[keys + ["predicted", "confidence"]].rename(
        columns={"predicted": "pred_m1", "confidence": "conf_m1"}
    )
    m2 = df_m2[keys + ["predicted", "confidence"]].rename(
        columns={"predicted": "pred_m2", "confidence": "conf_m2"}
    )

    paired = m1.merge(m2, on=keys, how="inner")
    print(f"Loaded {len(paired)} paired predictions "
          f"(M1: {len(df_m1)}, M2: {len(df_m2)})")
    return paired


# ── Confidence Extraction ─────────────────────────────────────────────────

_ANSWER_PAT = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_EXPLICIT_PAT = re.compile(
    r"\b(no\s+)?pneumothorax\s+detected\b", re.IGNORECASE
)


def extract_confidence(generated_text: str) -> float:
    """
    Estimate prediction confidence from the generated text.

    Levels:
      1.0 — Explicit label found in <answer> section
            (e.g., "PNEUMOTHORAX DETECTED" or "NO PNEUMOTHORAX DETECTED")
      0.7 — Keyword match in <answer> section (fallback pattern)
      0.5 — No <answer> tag, matched from full text
      0.3 — Default/unparsable (model was uncertain or output was malformed)
    """
    if not generated_text or not generated_text.strip():
        return 0.3

    m = _ANSWER_PAT.search(generated_text)
    if m:
        answer_text = m.group(1)
        # Explicit v2 label in answer section → highest confidence
        if _EXPLICIT_PAT.search(answer_text):
            return 1.0
        # Some keyword match in answer section
        if re.search(r"\bpneumothorax\b", answer_text, re.IGNORECASE):
            return 0.7
        if re.search(r"\bnormal\b|\bno finding\b|\bnegative\b",
                      answer_text, re.IGNORECASE):
            return 0.7
        # Answer tag present but unclear content
        return 0.5

    # No answer tag — fallback to full text
    if _EXPLICIT_PAT.search(generated_text):
        return 0.5
    return 0.3


# ── Aggregation Strategies ────────────────────────────────────────────────

def s1_unanimous(pred_m1: np.ndarray, pred_m2: np.ndarray, **kw) -> np.ndarray:
    """S1: PTX only if BOTH models predict PTX. Most conservative."""
    return ((pred_m1 == 1) & (pred_m2 == 1)).astype(int)


def s2_any(pred_m1: np.ndarray, pred_m2: np.ndarray, **kw) -> np.ndarray:
    """S2: PTX if EITHER model predicts PTX. Most sensitive."""
    return ((pred_m1 == 1) | (pred_m2 == 1)).astype(int)


def s3_detect_verify(pred_m1: np.ndarray, pred_m2: np.ndarray, **kw) -> np.ndarray:
    """
    S3: Detector-then-Verifier (sequential double-reading).

    If M1 (detector) says Normal → final = Normal (skip verification).
    If M1 (detector) says PTX → use M2 (verifier) as final answer.

    This mirrors clinical double-reading: the sensitive first reader
    flags cases, then the specific second reader decides.
    """
    # Start with M1's prediction
    result = pred_m1.copy()
    # Where M1 says PTX (1), override with M2's answer
    m1_positive = pred_m1 == 1
    result[m1_positive] = pred_m2[m1_positive]
    return result


def s4_pbr_weighted(
    pred_m1: np.ndarray,
    pred_m2: np.ndarray,
    pbr_m1: float = 1.0,
    pbr_m2: float = 1.0,
    **kw,
) -> np.ndarray:
    """
    S4: PBR-Guided Weighted voting.

    Weight each model inversely proportional to |PBR - 1|.
    Model closer to ideal PBR=1.0 gets higher weight.

    weighted_score = w1 * pred_m1 + w2 * pred_m2
    final = 1 if weighted_score > 0.5, else 0

    PBR values must be computed externally per (k_shot, test_set).
    """
    # Distance from ideal (PBR=1.0), with floor to avoid division by zero
    eps = 0.01
    d1 = max(abs(pbr_m1 - 1.0), eps)
    d2 = max(abs(pbr_m2 - 1.0), eps)

    # Inverse distance → higher weight for model closer to ideal
    w1_raw = 1.0 / d1
    w2_raw = 1.0 / d2
    total = w1_raw + w2_raw
    w1 = w1_raw / total
    w2 = w2_raw / total

    score = w1 * pred_m1.astype(float) + w2 * pred_m2.astype(float)
    return (score > 0.5).astype(int)


def s5_conf_weighted(
    pred_m1: np.ndarray,
    pred_m2: np.ndarray,
    conf_m1: np.ndarray | None = None,
    conf_m2: np.ndarray | None = None,
    **kw,
) -> np.ndarray:
    """
    S5: Confidence-Weighted voting.

    Weight each model by its per-sample confidence score.
    Uses answer-extraction certainty from extract_confidence().

    weighted_score = c1 * pred_m1 + c2 * pred_m2
    final = 1 if weighted_score > 0.5 * (c1 + c2), else 0
    """
    if conf_m1 is None:
        conf_m1 = np.ones_like(pred_m1, dtype=float)
    if conf_m2 is None:
        conf_m2 = np.ones_like(pred_m2, dtype=float)

    c1 = conf_m1.astype(float)
    c2 = conf_m2.astype(float)

    score = c1 * pred_m1.astype(float) + c2 * pred_m2.astype(float)
    threshold = 0.5 * (c1 + c2)
    return (score > threshold).astype(int)


# Strategy dispatch table
STRATEGY_FNS = {
    "S1_unanimous":     s1_unanimous,
    "S2_any":           s2_any,
    "S3_detect_verify": s3_detect_verify,
    "S4_pbr_weighted":  s4_pbr_weighted,
    "S5_conf_weighted": s5_conf_weighted,
}


# ── Bias Metrics ──────────────────────────────────────────────────────────

def compute_pbr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Prediction Bias Ratio = predicted_positive / actual_positive.

    PBR = 1.0 → ideal (balanced)
    PBR > 1.0 → over-predict
    PBR < 1.0 → under-predict
    """
    valid = y_pred != -1
    y_true_v = np.array(y_true)[valid]
    y_pred_v = np.array(y_pred)[valid]
    actual_pos = y_true_v.sum()
    if actual_pos == 0:
        return float("inf") if y_pred_v.sum() > 0 else 0.0
    return float(y_pred_v.sum()) / float(actual_pos)


def compute_sdr(spec_baseline: float, spec_k: float) -> float:
    """
    Specificity Drop Rate (%) = (spec_k0 - spec_k) / spec_k0 × 100.

    SDR = 0%   → no degradation
    SDR > 0%   → ICL degrades specificity
    SDR < 0%   → ICL improves specificity
    """
    if spec_baseline == 0:
        return 0.0
    return (spec_baseline - spec_k) / spec_baseline * 100.0


# ── PBR Lookup Table ──────────────────────────────────────────────────────

def compute_pbr_table(df_paired: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PBR for M1 and M2 per (k_shot, test_set), pooled over seeds.

    Returns DataFrame with columns:
        k_shot, test_set, pbr_m1, pbr_m2
    """
    rows = []
    for (k, ts), grp in df_paired.groupby(["k_shot", "test_set"]):
        gt = grp["ground_truth"].values
        pbr_m1 = compute_pbr(gt, grp["pred_m1"].values)
        pbr_m2 = compute_pbr(gt, grp["pred_m2"].values)
        rows.append({"k_shot": k, "test_set": ts, "pbr_m1": pbr_m1, "pbr_m2": pbr_m2})
    return pd.DataFrame(rows)


# ── Main Ensemble Evaluation ──────────────────────────────────────────────

def run_ensemble(
    df_paired: pd.DataFrame,
    strategies: list[str] | None = None,
    per_seed: bool = False,
) -> pd.DataFrame:
    """
    Run all (or selected) ensemble strategies on paired predictions.

    For S4, PBR values are computed per (k_shot, test_set) and passed
    automatically. For S5, per-sample confidence scores are used.

    Args:
        df_paired: Output of load_paired()
        strategies: List of strategy names (default: all 5)
        per_seed: If True, evaluate per seed. If False, pool seeds.

    Returns:
        DataFrame with columns:
            strategy, k_shot, test_set, [seed], sensitivity, specificity,
            precision, f1, accuracy, pbr, sdr, tp, tn, fp, fn, ...
    """
    if strategies is None:
        strategies = STRATEGIES

    # Pre-compute PBR table for S4
    pbr_table = compute_pbr_table(df_paired)
    pbr_lookup = {
        (row["k_shot"], row["test_set"]): (row["pbr_m1"], row["pbr_m2"])
        for _, row in pbr_table.iterrows()
    }

    # Group keys
    group_cols = ["k_shot", "test_set"]
    if per_seed:
        group_cols.append("seed")

    # Pre-compute baseline specificity (k=0) per test_set for SDR
    spec_baselines = {}  # (strategy, test_set) → spec at k=0

    all_rows = []

    for group_key, grp in df_paired.groupby(group_cols):
        if per_seed:
            k, ts, seed = group_key
        else:
            k, ts = group_key
            seed = "all"

        gt = grp["ground_truth"].values
        pred_m1 = grp["pred_m1"].values
        pred_m2 = grp["pred_m2"].values
        conf_m1 = grp["conf_m1"].values
        conf_m2 = grp["conf_m2"].values

        # Get PBR for this (k, ts) — used by S4
        pbr_m1_val, pbr_m2_val = pbr_lookup.get((k, ts), (1.0, 1.0))

        for strat_name in strategies:
            fn = STRATEGY_FNS[strat_name]
            ensemble_pred = fn(
                pred_m1, pred_m2,
                pbr_m1=pbr_m1_val, pbr_m2=pbr_m2_val,
                conf_m1=conf_m1, conf_m2=conf_m2,
            )

            metrics = compute_classification_metrics(gt, ensemble_pred)
            pbr = compute_pbr(gt, ensemble_pred)

            # Agreement between M1 and M2
            agree = (pred_m1 == pred_m2).sum()
            agree_rate = agree / len(pred_m1)

            # Conflict resolution: when M1 != M2, how often is ensemble correct?
            conflict_mask = pred_m1 != pred_m2
            if conflict_mask.sum() > 0:
                conflict_correct = (ensemble_pred[conflict_mask] == gt[conflict_mask]).sum()
                conflict_acc = conflict_correct / conflict_mask.sum()
            else:
                conflict_acc = None

            row = {
                "strategy": strat_name,
                "k_shot": k,
                "test_set": ts,
                "seed": seed,
                "pbr": round(pbr, 4),
                "agreement_rate": round(agree_rate, 4),
                "conflict_resolution_acc": round(conflict_acc, 4) if conflict_acc is not None else None,
                **metrics,
            }
            all_rows.append(row)

    result = pd.DataFrame(all_rows)

    # Compute SDR: need spec at k=0 as baseline per (strategy, test_set, [seed])
    sdr_group = ["strategy", "test_set"]
    if per_seed:
        sdr_group.append("seed")

    sdrs = []
    for idx, row in result.iterrows():
        key = tuple(row[c] for c in sdr_group)
        if row["k_shot"] == 0:
            spec_baselines[key] = row["specificity"]

    for idx, row in result.iterrows():
        key = tuple(row[c] for c in sdr_group)
        baseline = spec_baselines.get(key)
        if baseline is not None and row["specificity"] is not None:
            sdrs.append(round(compute_sdr(baseline, row["specificity"]), 2))
        else:
            sdrs.append(None)

    result["sdr"] = sdrs

    sort_cols = ["test_set", "strategy", "k_shot"]
    if per_seed:
        sort_cols.append("seed")
    return result.sort_values(sort_cols).reset_index(drop=True)


# ── Single-Model Baselines ───────────────────────────────────────────────

def run_single_model_baselines(
    df_paired: pd.DataFrame,
    per_seed: bool = False,
) -> pd.DataFrame:
    """
    Compute single-model metrics for M1 and M2 as comparison baselines.
    Uses the same grouped structure as run_ensemble().
    """
    group_cols = ["k_shot", "test_set"]
    if per_seed:
        group_cols.append("seed")

    spec_baselines = {}
    all_rows = []

    for group_key, grp in df_paired.groupby(group_cols):
        if per_seed:
            k, ts, seed = group_key
        else:
            k, ts = group_key
            seed = "all"

        gt = grp["ground_truth"].values

        for model_label, pred_col in [("M1_single", "pred_m1"), ("M2_single", "pred_m2")]:
            pred = grp[pred_col].values
            metrics = compute_classification_metrics(gt, pred)
            pbr = compute_pbr(gt, pred)

            row = {
                "strategy": model_label,
                "k_shot": k,
                "test_set": ts,
                "seed": seed,
                "pbr": round(pbr, 4),
                "agreement_rate": None,
                "conflict_resolution_acc": None,
                **metrics,
            }
            all_rows.append(row)

    result = pd.DataFrame(all_rows)

    # SDR
    sdr_group = ["strategy", "test_set"]
    if per_seed:
        sdr_group.append("seed")

    for idx, row in result.iterrows():
        key = tuple(row[c] for c in sdr_group)
        if row["k_shot"] == 0:
            spec_baselines[key] = row["specificity"]

    sdrs = []
    for idx, row in result.iterrows():
        key = tuple(row[c] for c in sdr_group)
        baseline = spec_baselines.get(key)
        if baseline is not None and row["specificity"] is not None:
            sdrs.append(round(compute_sdr(baseline, row["specificity"]), 2))
        else:
            sdrs.append(None)
    result["sdr"] = sdrs

    sort_cols = ["test_set", "strategy", "k_shot"]
    if per_seed:
        sort_cols.append("seed")
    return result.sort_values(sort_cols).reset_index(drop=True)


# ── Comparison Helper ─────────────────────────────────────────────────────

def compare_all(
    df_paired: pd.DataFrame,
    per_seed: bool = False,
) -> pd.DataFrame:
    """
    Run ensemble strategies + single-model baselines and return combined DataFrame.
    Makes it easy to compare ensemble vs single model performance.
    """
    df_ens = run_ensemble(df_paired, per_seed=per_seed)
    df_base = run_single_model_baselines(df_paired, per_seed=per_seed)
    return pd.concat([df_base, df_ens], ignore_index=True).sort_values(
        ["test_set", "k_shot", "strategy"]
    ).reset_index(drop=True)


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run dual-agent ensemble evaluation")
    parser.add_argument("--m1", default="results-exp-1-v2/M1/all_results.jsonl")
    parser.add_argument("--m2", default="results-exp-1-v2/M2/all_results.jsonl")
    parser.add_argument("--per_seed", action="store_true")
    parser.add_argument("--output", default=None, help="Save results to CSV")
    args = parser.parse_args()

    df_paired = load_paired(args.m1, args.m2)
    df_results = compare_all(df_paired, per_seed=args.per_seed)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df_results.to_string(index=False))

    if args.output:
        df_results.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
