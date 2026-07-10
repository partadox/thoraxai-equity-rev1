# Corrected-data (data2) results — consolidated

All from a SINGLE consistent source + correct format-aware parser (parse_dx.py).
Stage-1: M1/M3/M4 @10 seeds. During-inference: v3, k in {0,1,4,8} x 5 seeds.

## Table 3 — individual F1 (internal, k0->k8)
M1 0.338->0.660 | M3 ~0.01 (unresponsive) | M4 0.438->0.700   (external: M1 0.54->0.81, M4 0.83->0.92)

## Table 4 — bias (internal, k8): M1 PBR 1.41 SDR 61% | M4 PBR 1.03 SDR 24%

## Table 5 — after-inference (best-k, F1 int/ext)
M1 0.660/0.812 | M4 0.700/0.921 | S1 0.649/0.890 | S2-OR 0.720/0.898 | S3 0.649/0.890 | S4 0.700/0.916 | S5 0.659/0.813
OR-gate monotonicity at equal k=8 HOLDS (S2 sens 0.928>=max, spec 0.273<=min) -> resolves Reviewer 1.

## Table 6 — during-inference v3 (best-k, F1 int/ext)
M1 0.674/0.833 | D1 0.672/0.865 | D2 0.672/0.865 | D3 0.673/0.865 | D4-OR 0.691/0.876

## Table 7 — after vs during (central)
After S2-OR 0.720/0.898  >  During D4-OR 0.691/0.876  -> after-inference beats during (paper claim SURVIVES).

## Confidence calibration (M4 reviewer): mean 94.0% std 4.3%; correct 94.2% vs incorrect 93.5%.
NOTE: paper claimed 'higher when WRONG' (94.0 vs 93.1) -> did NOT replicate on corrected data (now marginally higher when correct). Still uninformative (tiny separation). Reframe this claim.

## KEY NARRATIVE SHIFTS (honest)
1. M4 alone is now very strong (0.700/0.921) -> ensemble gain marginal internally (+0.02) and NEGATIVE externally (M4 0.921 > S2 0.898). Reframe 'ensemble wins'.
2. After>During confirmed. OR-gate monotonicity resolved.
3. Confidence 'higher when wrong' NOT replicated.
4. 948 v1-contradictions was a v1-biased-prompt phenomenon; v1 during-inference NOT re-run.
