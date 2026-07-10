"""
Prompt builder for chest X-ray VLM inference.

Supports two prompt versions:
  - v1: Original (pneumothorax-focused, used in Exp1 v1)
  - v2: Balanced (no pneumothorax bias, used in Exp1/Exp2 v2)

Design rationale:
  Visual image-assistant pairs as few-shot demonstrations cause models
  to copy example outputs rather than analyse the query image independently.
  Instead, few-shot examples are embedded as text descriptions of reference cases
  (diagnosis label + truncated impression) before the query image.
  This preserves independent visual reasoning while providing contextual priors
  from the CheXpert Plus demonstration pool.

k_shot parameter counts examples PER CLASS (positive + negative).
Total text examples shown = 2 × k_shot.
"""

import re
from pathlib import Path
from PIL import Image


# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert radiologist assistant specialized in chest X-ray (CXR) analysis, "
    "with particular expertise in pneumothorax detection.\n\n"
    "For each chest X-ray, provide:\n"
    "1. SYSTEMATIC REVIEW: Evaluate lungs, heart, mediastinum, bones, soft tissues\n"
    "2. FINDINGS: Describe all abnormalities or confirm normal appearance\n"
    "3. IMPRESSION: Primary diagnosis — state PNEUMOTHORAX DETECTED or NO PNEUMOTHORAX\n"
    "4. RECOMMENDATION: Suggested clinical action\n\n"
    "If pneumothorax is present, describe location (left/right/bilateral), "
    "estimated size, lung collapse, and tension signs."
)

# ── Per-image user prompt ──────────────────────────────────────────────────
USER_QUERY_TEXT = "Analyze this chest X-ray for pneumothorax."


# ══════════════════════════════════════════════════════════════════════════
# PROMPT V2 — Balanced (no pneumothorax bias)
# Changes from v1:
#   - Removed "particular expertise in pneumothorax detection"
#   - Added "A normal CXR should be classified as NO PNEUMOTHORAX DETECTED"
#   - User query: "Analyze this chest X-ray." (no "for pneumothorax")
#   - Few-shot: "Evaluate independently" instruction added
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_V2 = (
    "You are an expert radiologist assistant specialized in chest X-ray (CXR) analysis.\n\n"
    "For each chest X-ray, provide:\n"
    "1. SYSTEMATIC REVIEW: Evaluate lungs, heart, mediastinum, bones, soft tissues\n"
    "2. FINDINGS: Describe all abnormalities or confirm normal appearance\n"
    "3. IMPRESSION: Primary diagnosis with your assessment\n"
    "4. CLASSIFICATION: State exactly one of: PNEUMOTHORAX DETECTED or NO PNEUMOTHORAX DETECTED\n\n"
    "Important: Base your classification strictly on radiographic evidence. "
    "A normal chest X-ray should be classified as NO PNEUMOTHORAX DETECTED."
)

USER_QUERY_V2 = "Analyze this chest X-ray."


# ── Report cleaning ────────────────────────────────────────────────────────
_BOILERPLATE_PAT = re.compile(
    r"I have personally reviewed\b.*?(?:\.\s|$)",
    re.IGNORECASE | re.DOTALL,
)
_MAX_REPORT_WORDS = 60


def _clean_report(text: str) -> str:
    """Clean and truncate a radiology impression for context embedding."""
    text = _BOILERPLATE_PAT.sub("", text)
    text = re.sub(r"[\x00-\x1f\r\n]+", " ", text).strip()
    words = text.split()
    if len(words) > _MAX_REPORT_WORDS:
        text = " ".join(words[:_MAX_REPORT_WORDS]) + "..."
    return text


def _build_few_shot_context(few_shot_examples: list[dict], k_shot: int) -> str:
    """
    Build text-based context block from few-shot examples.

    Returns a string like:
        REFERENCE CASES FROM CHEXPERT PLUS (n=4):

        Case 1 (PNEUMOTHORAX): Small right apical pneumothorax...
        Case 2 (NO PNEUMOTHORAX): No acute cardiopulmonary process...
        ...

        Use these reference cases to calibrate your analysis of the image below.
    """
    positives = [ex for ex in few_shot_examples if ex["category"] == "positive"]
    negatives = [ex for ex in few_shot_examples if ex["category"] == "negative"]
    n = min(k_shot, len(positives), len(negatives))
    if n == 0:
        return ""

    parts = [f"REFERENCE CASES FROM CHEXPERT PLUS (n={2 * n}):"]
    case_num = 1
    for i in range(n):
        ptx_report  = _clean_report(positives[i]["report"])
        norm_report = _clean_report(negatives[i]["report"])
        parts.append(f"\nCase {case_num} (PNEUMOTHORAX): {ptx_report}")
        case_num += 1
        parts.append(f"Case {case_num} (NO PNEUMOTHORAX): {norm_report}")
        case_num += 1

    parts.append(
        "\nUse these reference cases to calibrate your analysis of the chest X-ray below."
    )
    return "\n".join(parts)


def build_few_shot_context_v2(few_shot_examples: list[dict], k_shot: int) -> str:
    """
    Build text-based few-shot context block for prompt v2 (balanced).

    Differences from v1 (_build_few_shot_context):
      - Header says "REFERENCE CASES" (no "FROM CHEXPERT PLUS")
      - Labels use full form: "PNEUMOTHORAX DETECTED" / "NO PNEUMOTHORAX DETECTED"
      - Adds "Evaluate the image independently" instruction

    Returns empty string if k_shot == 0.
    """
    positives = [ex for ex in few_shot_examples if ex["category"] == "positive"]
    negatives = [ex for ex in few_shot_examples if ex["category"] == "negative"]
    n = min(k_shot, len(positives), len(negatives))
    if n == 0:
        return ""

    parts = [f"REFERENCE CASES (n={2 * n}):"]
    case_num = 1
    for i in range(n):
        ptx_report  = _clean_report(positives[i]["report"])
        norm_report = _clean_report(negatives[i]["report"])
        parts.append(f"\nCase {case_num} (PNEUMOTHORAX DETECTED): {ptx_report}")
        case_num += 1
        parts.append(f"Case {case_num} (NO PNEUMOTHORAX DETECTED): {norm_report}")
        case_num += 1

    parts.append(
        "\nUse these reference cases to calibrate your analysis of the chest X-ray below. "
        "Evaluate the image independently — do not assume the query matches any reference case."
    )
    return "\n".join(parts)


def _load_image(image_path: str | Path) -> Image.Image:
    """Load image from disk as PIL Image (RGB)."""
    img = Image.open(str(image_path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def build_messages(
    few_shot_examples: list[dict],
    query_image_path: str | Path,
    k_shot: int,
    load_images: bool = True,
) -> list[dict]:
    """
    Build chat messages for NV-Reason-CXR-3B inference.

    Args:
        few_shot_examples : list of dicts from FewShotSelector.select()
                            keys: image_path, report, label, category, file_id
        query_image_path  : path to the test image (PNG)
        k_shot            : number of examples per class to embed as text context
                            (0 = zero-shot, i.e. no reference cases)
        load_images       : if True, load query image as PIL.Image (for inference);
                            set False only for unit testing

    Returns:
        list of message dicts in Qwen2.5-VL chat format
        Structure:
          [system]
          [user: (optional text context) + image + query_text]

        Total messages: always 2 (system + one user turn).
        Only the query image is passed; few-shot examples are text-only.
    """
    messages = []

    # ── System message ──────────────────────────────────────────────────────
    messages.append({
        "role"   : "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    })

    # ── User turn ───────────────────────────────────────────────────────────
    query_img = (
        _load_image(query_image_path) if load_images
        else str(query_image_path)
    )

    content: list[dict] = []

    # Prepend text context if k_shot > 0
    if k_shot > 0 and few_shot_examples:
        ctx = _build_few_shot_context(few_shot_examples, k_shot)
        if ctx:
            content.append({"type": "text", "text": ctx})

    # Query image + prompt
    content.append({"type": "image", "image": query_img})
    content.append({"type": "text",  "text" : USER_QUERY_TEXT})

    messages.append({"role": "user", "content": content})

    return messages


def describe_messages(messages: list[dict]) -> str:
    """Return a human-readable summary of the message structure (for debugging)."""
    lines = []
    for i, msg in enumerate(messages):
        role    = msg["role"]
        content = msg["content"]
        types   = [c["type"] for c in content] if isinstance(content, list) else [type(content).__name__]
        text_preview = ""
        for c in (content if isinstance(content, list) else []):
            if c["type"] == "text":
                text_preview = c["text"][:60].replace("\n", " ")
                break
        lines.append(f"  [{i}] {role:<10} {types}  '{text_preview}...'")
    return "\n".join(lines)
