from __future__ import annotations

"""
During-Inference Ensemble: M1→M2 Sequential (Detector→Reviewer).

Instead of running M1 and M2 independently and voting after the fact,
this pipeline feeds M1's reasoning output INTO M2's prompt so M2 can
act as an informed reviewer rather than a blind second reader.

Architecture:
    CXR Image → M1 (Detector) → reasoning + prediction
                                        ↓
    CXR Image + M1's reasoning → M2 (Reviewer) → final decision

Key advantage: M2's process-factuality training is leveraged to EVALUATE
M1's claims, not detect from scratch. This should unlock M2's specificity
advantage without sacrificing M1's sensitivity.

M1 data is REUSED from existing Exp1 v2 results — only M2 needs new inference.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ── System Prompt for M2 as Reviewer ─────────────────────────────────────

# ── v1 prompt (DEPRECATED — caused always-NO bias, sensitivity=0.0) ──
# Keeping for reference. The instructions "CORRECT any over-detection" and
# "if ambiguous → NO" made M4 default to NO for 99.97% of cases.
SYSTEM_PROMPT_REVIEWER_V1 = (
    "You are an expert radiologist assistant performing a second reading of a chest X-ray (CXR).\n\n"
    "A first reader (AI model) has already analyzed this image. "
    "Their full analysis is provided below.\n\n"
    "Your role:\n"
    "1. REVIEW the first reader's analysis — check if their observations match the radiographic evidence\n"
    "2. VERIFY key findings — look at the specific areas the first reader mentions\n"
    "3. CORRECT any over-detection — the first reader tends to be over-sensitive and may report "
    "pneumothorax when the image is actually normal\n"
    "4. CLASSIFICATION: State exactly one of: PNEUMOTHORAX DETECTED or NO PNEUMOTHORAX DETECTED\n\n"
    "Important:\n"
    "- Base your classification strictly on radiographic evidence visible in the image.\n"
    "- If the first reader claims pneumothorax but the evidence is ambiguous or absent, "
    "classify as NO PNEUMOTHORAX DETECTED.\n"
    "- A normal chest X-ray should be classified as NO PNEUMOTHORAX DETECTED."
)

# ── v2 prompt (balanced — no directional bias) ──
SYSTEM_PROMPT_REVIEWER = (
    "You are an expert radiologist assistant performing a second reading of a chest X-ray (CXR).\n\n"
    "A first reader (AI model) has already analyzed this image. "
    "Their full analysis is provided below.\n\n"
    "Your role:\n"
    "1. REVIEW the first reader's analysis and check if their observations match what you see in the image\n"
    "2. VERIFY specific findings — if the first reader identifies pneumothorax signs "
    "(e.g., visceral pleural line, absent lung markings, hyperlucency), look carefully at those areas\n"
    "3. Make YOUR OWN independent classification based on the radiographic evidence\n\n"
    "CLASSIFICATION: State exactly one of: PNEUMOTHORAX DETECTED or NO PNEUMOTHORAX DETECTED\n\n"
    "Important:\n"
    "- Base your classification on radiographic evidence visible in the image.\n"
    "- If you see signs consistent with pneumothorax (even subtle), classify as PNEUMOTHORAX DETECTED.\n"
    "- If the lung fields are clear with no evidence of pneumothorax, classify as NO PNEUMOTHORAX DETECTED."
)

USER_QUERY_REVIEWER = (
    "Based on the first reader's analysis above and your own examination of this chest X-ray, "
    "provide your second reading assessment."
)


# ── Build M2 Reviewer Prompt ─────────────────────────────────────────────

def build_reviewer_user_text(
    m1_output: str,
    few_shot_context: str = "",
) -> str:
    """
    Build the user text content for M2's reviewer prompt.

    Includes:
    - Few-shot context (if k > 0) — same as M1 received
    - M1's full generated output (reasoning + answer)
    - Instruction for second reading
    """
    parts = []

    if few_shot_context:
        parts.append(few_shot_context)
        parts.append("")  # blank line separator

    parts.append("FIRST READER'S ANALYSIS:")
    parts.append(m1_output.strip())

    return "\n".join(parts)


def build_few_shot_context_v2(few_shot_examples: list[dict], k_shot: int) -> str:
    """
    Build text-based few-shot context (v2 balanced).
    Copied from prompt_builder.py for self-containment.
    """
    _BOILERPLATE_PAT = re.compile(
        r"I have personally reviewed\b.*?(?:\.\s|$)", re.IGNORECASE | re.DOTALL
    )
    MAX_WORDS = 60

    def clean_report(text):
        text = _BOILERPLATE_PAT.sub("", text)
        text = re.sub(r"[\x00-\x1f\r\n]+", " ", text).strip()
        words = text.split()
        if len(words) > MAX_WORDS:
            text = " ".join(words[:MAX_WORDS]) + "..."
        return text

    positives = [ex for ex in few_shot_examples if ex["category"] == "positive"]
    negatives = [ex for ex in few_shot_examples if ex["category"] == "negative"]
    n = min(k_shot, len(positives), len(negatives))
    if n == 0:
        return ""

    parts = [f"REFERENCE CASES (n={2 * n}):"]
    case_num = 1
    for i in range(n):
        ptx_report = clean_report(positives[i]["report"])
        norm_report = clean_report(negatives[i]["report"])
        parts.append(f"\nCase {case_num} (PNEUMOTHORAX DETECTED): {ptx_report}")
        case_num += 1
        parts.append(f"Case {case_num} (NO PNEUMOTHORAX DETECTED): {norm_report}")
        case_num += 1

    parts.append(
        "\nUse these reference cases to calibrate your analysis of the chest X-ray below. "
        "Evaluate the image independently — do not assume the query matches any reference case."
    )
    return "\n".join(parts)


# ── Load M1 Results ──────────────────────────────────────────────────────

def load_m1_results(jsonl_path: str | Path) -> pd.DataFrame:
    """Load M1's existing results from Exp1 v2."""
    records = [json.loads(line) for line in open(jsonl_path, encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
    print(f"Loaded M1 results: {len(df)} records")
    print(f"  k_shots: {sorted(df['k_shot'].unique())}")
    print(f"  seeds: {sorted(df['seed'].unique())}")
    print(f"  test_sets: {df['test_set'].unique().tolist()}")
    return df


def get_m1_output(
    df_m1: pd.DataFrame,
    k_shot: int,
    seed: int,
    test_set: str,
    image_id: str,
) -> str | None:
    """Get M1's generated_text for a specific (k, seed, test_set, image_id)."""
    mask = (
        (df_m1["k_shot"] == k_shot)
        & (df_m1["seed"] == seed)
        & (df_m1["test_set"] == test_set)
        & (df_m1["image_id"] == image_id)
    )
    matches = df_m1[mask]
    if len(matches) == 0:
        return None
    return matches.iloc[0]["generated_text"]


# ── Checkpoint Management ────────────────────────────────────────────────

def load_checkpoint(path: str | Path) -> tuple[set, list[dict]]:
    """Load existing checkpoint. Returns (done_keys, records)."""
    path = Path(path)
    done_keys = set()
    records = []
    if path.exists():
        for line in open(path, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                key = (rec["k_shot"], rec["seed"], rec["test_set"], rec["image_id"])
                done_keys.add(key)
                records.append(rec)
    return done_keys, records


def append_jsonl(path: str | Path, records: list[dict]):
    """Append records to JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
