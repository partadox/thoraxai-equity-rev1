"""
Few-shot example selector for NV-Reason-CXR-3B experiments.

Supports three strategies:
  - random    : random k per class (repeated with multiple seeds)
  - diversity : evenly spaced across report-length distribution
  - similarity: demographic matching (age + sex) to test-set distribution

Pool CSV columns expected:
  file_id, label, age, sex, section_impression, ...
  (from data/chexpert_ptx_positive.csv / chexpert_ptx_negative.csv)

Image paths are resolved as:
  {images_root}/{label}/{file_id}.png
  e.g.  data/chexpert_plus/images/pneumothorax/patient08809_study2_view1_frontal.png
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Path to PNG images root (relative to project root)
_DEFAULT_IMAGES_ROOT = Path("data/chexpert_plus/images")


class FewShotSelector:
    """Select k-shot examples from CheXpert Plus pool for each class."""

    def __init__(
        self,
        positive_csv: str,
        negative_csv: str,
        images_root: str | Path = _DEFAULT_IMAGES_ROOT,
    ):
        self.positives  = pd.read_csv(positive_csv, encoding="utf-8-sig")
        self.negatives  = pd.read_csv(negative_csv, encoding="utf-8-sig")
        self.images_root = Path(images_root)

        # Pre-compute report length for diversity strategy
        self.positives["_report_len"] = (
            self.positives["section_impression"].fillna("").apply(len)
        )
        self.negatives["_report_len"] = (
            self.negatives["section_impression"].fillna("").apply(len)
        )

        # Filter out rows with missing images
        self.positives = self._filter_existing(self.positives, "pneumothorax")
        self.negatives = self._filter_existing(self.negatives, "normal")

        print(
            f"[FewShotSelector] Pool loaded — "
            f"PTX: {len(self.positives)}, Normal: {len(self.negatives)}"
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _img_path(self, file_id: str, label: str) -> Path:
        """Resolve PNG path for a given file_id and label."""
        subfolder = "pneumothorax" if label == "pneumothorax" else "normal"
        return self.images_root / subfolder / f"{file_id}.png"

    def _filter_existing(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Keep only rows whose PNG file exists on disk."""
        exists = df["file_id"].apply(
            lambda fid: self._img_path(fid, label).exists()
        )
        missing = (~exists).sum()
        if missing:
            print(f"  [WARNING] {missing} {label} PNG files missing — excluded from pool")
        return df[exists].copy()

    def _format_examples(
        self, pos_df: pd.DataFrame, neg_df: pd.DataFrame
    ) -> list[dict]:
        """Return list of example dicts ready for prompt_builder."""
        examples = []
        for _, row in pos_df.iterrows():
            examples.append({
                "image_path" : str(self._img_path(row["file_id"], "pneumothorax")),
                "report"     : str(row["section_impression"]).strip(),
                "label"      : "pneumothorax",
                "category"   : "positive",
                "file_id"    : row["file_id"],
            })
        for _, row in neg_df.iterrows():
            examples.append({
                "image_path" : str(self._img_path(row["file_id"], "normal")),
                "report"     : str(row["section_impression"]).strip(),
                "label"      : "normal",
                "category"   : "negative",
                "file_id"    : row["file_id"],
            })
        return examples

    # ── Strategy 1: Random ─────────────────────────────────────────────────

    def select_random(self, k: int, seed: int = 42) -> list[dict]:
        """
        Random selection: k examples per class.
        With fixed seed for reproducibility; different seeds give different sets.
        """
        rng = np.random.RandomState(seed)
        pos = self.positives.sample(n=min(k, len(self.positives)), random_state=rng)
        neg = self.negatives.sample(n=min(k, len(self.negatives)), random_state=rng)
        return self._format_examples(pos, neg)

    # ── Strategy 2: Diversity ──────────────────────────────────────────────

    def select_diversity(self, k: int, seed: int = 42) -> list[dict]:
        """
        Diversity-based: maximize coverage by evenly sampling across the
        report-length distribution (proxy for clinical complexity).

        Algorithm:
          - Sort pool by section_impression length
          - Pick k evenly-spaced indices (linspace) to cover full range
        """
        def _pick(df: pd.DataFrame, n: int) -> pd.DataFrame:
            sorted_df = df.sort_values("_report_len").reset_index(drop=True)
            indices   = np.linspace(0, len(sorted_df) - 1, n, dtype=int)
            return sorted_df.iloc[indices]

        pos = _pick(self.positives, min(k, len(self.positives)))
        neg = _pick(self.negatives, min(k, len(self.negatives)))
        return self._format_examples(pos, neg)

    # ── Strategy 3: Similarity ─────────────────────────────────────────────

    def select_similarity(
        self,
        k: int,
        test_metadata: pd.DataFrame | None = None,
        seed: int = 42,
    ) -> list[dict]:
        """
        Similarity-based: select pool examples whose demographics best match
        the test-set distribution.

        Matching features:
          - Age  : pick examples closest to test-set mean age
          - Sex  : honour test-set sex ratio (majority first)

        Falls back to diversity if test_metadata is None or has no age column.
        """
        if test_metadata is None or "age" not in test_metadata.columns:
            print("[FewShotSelector] select_similarity: no test_metadata — falling back to diversity")
            return self.select_diversity(k, seed)

        test_age_mean = test_metadata["age"].dropna().mean()
        if np.isnan(test_age_mean):
            return self.select_diversity(k, seed)

        # Determine dominant sex in test set (for ordering)
        if "sex" in test_metadata.columns:
            sex_counts   = test_metadata["sex"].value_counts()
            dominant_sex = sex_counts.index[0] if len(sex_counts) > 0 else None
        else:
            dominant_sex = None

        def _pick_similar(df: pd.DataFrame, n: int) -> pd.DataFrame:
            df = df.copy()
            df["_age_diff"] = (df["age"].fillna(test_age_mean) - test_age_mean).abs()

            if dominant_sex and "sex" in df.columns:
                # Sort: dominant sex first, then by age similarity
                df["_sex_rank"] = (df["sex"] != dominant_sex).astype(int)
                df = df.sort_values(["_sex_rank", "_age_diff"])
            else:
                df = df.sort_values("_age_diff")

            return df.head(n)

        pos = _pick_similar(self.positives, min(k, len(self.positives)))
        neg = _pick_similar(self.negatives, min(k, len(self.negatives)))
        return self._format_examples(pos, neg)

    # ── Unified entry point ────────────────────────────────────────────────

    def select(
        self,
        strategy: str,
        k: int,
        seed: int = 42,
        test_metadata: pd.DataFrame | None = None,
    ) -> list[dict]:
        """
        Dispatch to the appropriate strategy.

        Args:
            strategy      : "random" | "diversity" | "similarity"
            k             : number of examples per class
            seed          : random seed (used by random & diversity tiebreaking)
            test_metadata : DataFrame with test-set demographics (for similarity)

        Returns:
            list of example dicts with keys:
              image_path, report, label, category, file_id
        """
        if k == 0:
            return []
        if strategy == "random":
            return self.select_random(k, seed)
        elif strategy == "diversity":
            return self.select_diversity(k, seed)
        elif strategy == "similarity":
            return self.select_similarity(k, test_metadata, seed)
        else:
            raise ValueError(f"Unknown strategy: '{strategy}'. Use 'random', 'diversity', or 'similarity'.")
