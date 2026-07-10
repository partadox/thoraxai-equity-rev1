"""
Evaluation utilities for NV-Reason-CXR-3B pneumothorax detection experiments.

The model outputs text in the format:
    <think> ... </think>
    <answer> Finding1, Finding2, ... </answer>

Parsing:
    extract_diagnosis() → 1 (pneumothorax detected) or 0 (not detected)
    The "<answer>" section is searched for the word "pneumothorax" (case-insensitive).

Metrics computed per (k_shot, strategy, seed, test_set) group:
    - sensitivity (recall / true positive rate)
    - specificity (true negative rate)
    - precision
    - f1
    - accuracy
    - n_pos, n_neg, tp, tn, fp, fn
"""

import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# ── Answer extraction ──────────────────────────────────────────────────────

_ANSWER_PAT  = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_PTX_PAT     = re.compile(r"\bpneumothorax\b", re.IGNORECASE)


def extract_diagnosis(generated_text: str) -> int:
    """
    Extract binary pneumothorax prediction from model output.

    Steps:
      1. Try to parse the <answer>...</answer> section.
      2. If no <answer> tag, search the entire text.
      3. Return 1 if "pneumothorax" is found, else 0.

    Args:
        generated_text : raw string from NVReasonCXREngine.generate()

    Returns:
        1  (pneumothorax predicted present)
        0  (pneumothorax predicted absent / uncertain / no finding)
       -1  (empty / unparsable output)
    """
    if not generated_text or not generated_text.strip():
        return -1

    m = _ANSWER_PAT.search(generated_text)
    search_text = m.group(1) if m else generated_text

    return 1 if _PTX_PAT.search(search_text) else 0


# ── Classification metrics ─────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> dict:
    """
    Compute binary classification metrics.

    Ignores samples where y_pred == -1 (unparsable).

    Returns dict with keys:
        sensitivity, specificity, precision, f1, accuracy,
        n_pos, n_neg, tp, tn, fp, fn, n_invalid, n_total
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Filter unparsable
    valid   = y_pred != -1
    n_invalid = int((~valid).sum())
    y_true_v  = y_true[valid]
    y_pred_v  = y_pred[valid]

    if len(y_true_v) == 0:
        return {k: None for k in
                ["sensitivity", "specificity", "precision", "f1", "accuracy",
                 "n_pos", "n_neg", "tp", "tn", "fp", "fn"]} | {
            "n_invalid": n_invalid, "n_total": len(y_true)
        }

    tn, fp, fn, tp = confusion_matrix(y_true_v, y_pred_v, labels=[0, 1]).ravel()
    tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)
    n_pos = int(y_true_v.sum())
    n_neg = int(len(y_true_v) - n_pos)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy    = (tp + tn) / len(y_true_v)

    return {
        "sensitivity" : round(sensitivity, 4),
        "specificity" : round(specificity, 4),
        "precision"   : round(precision, 4),
        "f1"          : round(f1, 4),
        "accuracy"    : round(accuracy, 4),
        "n_pos"       : n_pos,
        "n_neg"       : n_neg,
        "tp"          : tp,
        "tn"          : tn,
        "fp"          : fp,
        "fn"          : fn,
        "n_invalid"   : n_invalid,
        "n_total"     : len(y_true),
    }


# ── Aggregate over a results DataFrame ────────────────────────────────────

def evaluate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate all experiment results from a DataFrame of inference records.

    Expected columns: k_shot, strategy, seed, test_set, ground_truth, generated_text
    Adds column: predicted (0/1/-1)
    Returns aggregated metrics per (k_shot, strategy, test_set).

    The aggregation over seeds averages sensitivity, specificity, f1, accuracy.
    """
    df = df.copy()
    df["predicted"] = df["generated_text"].apply(extract_diagnosis)

    rows = []
    for (k, strat, ts), grp in df.groupby(["k_shot", "strategy", "test_set"]):
        # Aggregate over seeds: pool all predictions
        metrics = compute_classification_metrics(
            grp["ground_truth"].tolist(),
            grp["predicted"].tolist(),
        )
        rows.append({
            "k_shot"     : k,
            "strategy"   : strat,
            "test_set"   : ts,
            "n_seeds"    : grp["seed"].nunique(),
            **metrics,
        })

    return pd.DataFrame(rows).sort_values(["test_set", "k_shot", "strategy"])


def evaluate_results_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Like evaluate_results() but keeps per-seed rows (for variance analysis).
    """
    df = df.copy()
    df["predicted"] = df["generated_text"].apply(extract_diagnosis)

    rows = []
    for (k, strat, seed, ts), grp in df.groupby(
        ["k_shot", "strategy", "seed", "test_set"]
    ):
        metrics = compute_classification_metrics(
            grp["ground_truth"].tolist(),
            grp["predicted"].tolist(),
        )
        rows.append({
            "k_shot"  : k,
            "strategy": strat,
            "seed"    : seed,
            "test_set": ts,
            **metrics,
        })

    return pd.DataFrame(rows).sort_values(
        ["test_set", "k_shot", "strategy", "seed"]
    )


# ── CLI: evaluate a checkpoint JSONL ──────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate experiment results from a JSONL checkpoint."
    )
    parser.add_argument(
        "jsonl_path", nargs="?",
        default="results/all_results.jsonl",
        help="Path to JSONL results file",
    )
    parser.add_argument(
        "--per_seed", action="store_true",
        help="Show per-seed metrics instead of aggregated",
    )
    args = parser.parse_args()

    path = Path(args.jsonl_path)
    if not path.exists():
        print(f"File not found: {path}")
        raise SystemExit(1)

    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from {path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Test sets: {df['test_set'].unique().tolist()}")
    print(f"k_shots: {sorted(df['k_shot'].unique().tolist())}")
    print()

    if args.per_seed:
        result_df = evaluate_results_per_seed(df)
    else:
        result_df = evaluate_results(df)

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(result_df.to_string(index=False))
