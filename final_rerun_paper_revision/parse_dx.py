"""
Format-aware pneumothorax verdict parser (use this for Stage-2 analysis).

WHY: the old src/evaluate.extract_diagnosis returns 1 whenever the word
"pneumothorax" appears, but prose outputs (M3/M4) say "NO PNEUMOTHORAX DETECTED"
-> wrongly counted positive (degenerate all-positive F1=0.667). This parser:
  - M1 (NV-Reason): reads the <answer> finding-list; positive iff "pneumothorax" listed.
  - M3/M4 (prose):  reads the CLASSIFICATION verdict, negation-aware.
Returns 1 (PTX), 0 (no PTX), -1 (empty/unparsable).
"""
import re
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)

def parse_diagnosis(text: str) -> int:
    if not text or not text.strip():
        return -1
    m = _ANSWER.search(text)
    if m:  # M1 finding-list answer (no negation phrasing)
        return 1 if re.search(r"\bpneumothorax\b", m.group(1), re.I) else 0
    t = text.upper()
    verdicts = re.findall(r"(NO\s+PNEUMOTHORAX\s+DETECTED|PNEUMOTHORAX\s+DETECTED)", t)
    if verdicts:
        return 0 if verdicts[-1].startswith("NO") else 1
    if re.search(r"NO\s+(EVIDENCE\s+OF\s+)?PNEUMOTHORAX|WITHOUT\s+PNEUMOTHORAX", t):
        return 0
    return 1 if "PNEUMOTHORAX" in t else 0
