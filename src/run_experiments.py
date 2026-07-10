"""
Main experiment runner for NV-Reason-CXR-3B few-shot pneumothorax detection.

Experiments:
  1. K-shot ablation  : k ∈ {0,1,2,4,8} × 5 seeds × 2 test sets
  2. Strategy comparison : {random, diversity, similarity} × 5 seeds × best_k × 2 test sets

Results saved as JSONL with checkpoint support (resumes if interrupted).

Usage:
    cd D:/1phd/thoraxai_research/1_equity
    source .venv/Scripts/activate
    python src/run_experiments.py
    python src/run_experiments.py --config configs/experiment_config.yaml
    python src/run_experiments.py --exp 2          # only Experiment 2
    python src/run_experiments.py --best_k 4       # override best_k for Exp2
"""

import sys
import json
import argparse
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.example_selector import FewShotSelector
from src.prompt_builder import build_messages
from src.inference import NVReasonCXREngine


# ── Helpers ────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _append_jsonl(path: Path, records: list[dict]) -> None:
    """Append records to a JSONL file (create if missing)."""
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_done_keys(checkpoint_path: Path) -> set:
    """Return set of (k_shot, strategy, seed, test_set, image_id) already saved."""
    done = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["k_shot"], r["strategy"], r["seed"],
                              r["test_set"], r["image_id"]))
                except Exception:
                    pass
    return done


def _img_path_rsunair(image_id: str, label_int: int, data_root: Path) -> Path:
    """Derive RS UNAIR PNG path from image_id and integer label."""
    subfolder = "pneumothorax" if label_int == 1 else "normal"
    return data_root / "rs_unair" / "images" / subfolder / f"{image_id}.png"


def _img_path_chexpert_test(file_id: str, label_str: str, data_root: Path) -> Path:
    """Derive CheXpert internal test PNG path from file_id and string label."""
    subfolder = "internal_test_pneumothorax" if label_str == "pneumothorax" else "internal_test_normal"
    return data_root / "chexpert_plus" / "images" / subfolder / f"{file_id}.png"


# ── Load test sets ─────────────────────────────────────────────────────────

def load_test_sets(cfg: dict, data_root: Path) -> dict:
    """
    Load both test sets with image paths and labels.

    Returns dict:
        {
          "chexpert_internal": {"name", "rows": DataFrame, "images": [Path], "metadata": DataFrame},
          "rsunair_external" : {...}
        }
    """
    test_sets = {}

    # ── CheXpert Plus — Internal Test (200 images) ─────────────────────────
    chex_df = pd.read_csv(cfg["data"]["chexpert_internal_test_csv"], encoding="utf-8-sig")
    chex_df["label_int"] = chex_df["label"].map({"pneumothorax": 1, "normal": 0})
    chex_df["image_id"]  = chex_df["file_id"]   # alias for uniform access
    chex_images = [
        _img_path_chexpert_test(row["file_id"], row["label"], data_root)
        for _, row in chex_df.iterrows()
    ]
    missing_chex = sum(1 for p in chex_images if not p.exists())
    print(f"[TestSet] CheXpert Internal: {len(chex_df)} images, {missing_chex} PNG missing")

    test_sets["chexpert_internal"] = {
        "name"    : "CheXpert Plus (Internal)",
        "rows"    : chex_df.reset_index(drop=True),
        "images"  : chex_images,
        "metadata": chex_df,   # for similarity selection
    }

    # ── RS UNAIR — External Validation (278 images) ────────────────────────
    rs_labels = pd.read_csv(cfg["data"]["rsunair_labels_csv"], encoding="utf-8-sig")
    # Load full metadata for demographics (age, sex) — needed for similarity
    rs_meta = pd.read_csv(
        Path(cfg["data"]["rsunair_root"]) / "metadata_normalized_dicom.csv",
        encoding="utf-8-sig", sep=None, engine="python",
    )
    # Merge label into metadata on file_id = image_id
    rs_labels_merged = rs_labels.rename(columns={"image_id": "file_id", "label": "label_int"})
    rs_meta_merged = rs_meta.merge(rs_labels_merged, on="file_id", how="inner")
    rs_meta_merged["image_id"] = rs_meta_merged["file_id"]

    rs_images = [
        _img_path_rsunair(row["image_id"], row["label_int"], data_root)
        for _, row in rs_meta_merged.iterrows()
    ]
    missing_rs = sum(1 for p in rs_images if not p.exists())
    print(f"[TestSet] RS UNAIR External: {len(rs_meta_merged)} images, {missing_rs} PNG missing")

    test_sets["rsunair_external"] = {
        "name"    : "RS UNAIR (External)",
        "rows"    : rs_meta_merged.reset_index(drop=True),
        "images"  : rs_images,
        "metadata": rs_meta_merged,   # has age, sex for similarity
    }

    return test_sets


# ── Core: run on one test set ──────────────────────────────────────────────

def run_on_test_set(
    engine: NVReasonCXREngine,
    examples: list[dict],
    test_set: dict,
    k: int,
    strategy: str,
    seed: int,
    experiment_type: str,
    cfg: dict,
    done_keys: set,
) -> list[dict]:
    """
    Run inference on every image in test_set, skip already-done images.
    Returns list of new result records.
    """
    model_cfg = cfg["model"]
    records   = []

    for img_path, (_, row) in zip(test_set["images"], test_set["rows"].iterrows()):
        image_id  = str(row["image_id"])
        key       = (k, strategy, seed, test_set["name"], image_id)

        if key in done_keys:
            continue  # resume: already saved
        if not img_path.exists():
            print(f"  [SKIP] missing PNG: {img_path.name}")
            continue

        messages = build_messages(examples, img_path, k_shot=k, load_images=True)

        try:
            generated_text, inference_time = engine.generate(
                messages,
                max_new_tokens=model_cfg["max_new_tokens"],
                temperature=model_cfg["temperature"],
                top_p=model_cfg["top_p"],
            )
        except Exception as e:
            generated_text, inference_time = "", 0.0
            print(f"  [ERROR] {image_id}: {e}")

        records.append({
            "experiment_type": experiment_type,
            "k_shot"         : k,
            "strategy"       : strategy,
            "seed"           : seed,
            "test_set"       : test_set["name"],
            "image_id"       : image_id,
            "ground_truth"   : int(row["label_int"]),
            "generated_text" : generated_text,
            "inference_time" : round(inference_time, 3),
            "timestamp"      : datetime.now().isoformat(),
        })

    return records


# ── Experiment 1: K-Shot Ablation ─────────────────────────────────────────

def run_experiment1(engine, selector, test_sets, cfg, output_dir, done_keys):
    """K-shot ablation: k ∈ k_shots × seeds × test_sets (strategy=random)."""
    exp_cfg    = cfg["experiments"]
    ckpt_path  = output_dir / "checkpoint_exp1.jsonl"
    all_records = []

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: K-Shot Ablation (strategy=random)")
    print("=" * 60)

    for k in exp_cfg["k_shots"]:
        for seed in range(exp_cfg["num_random_seeds"]):
            exp_id = f"k={k} seed={seed}"
            print(f"\n--- {exp_id} ---")

            examples = selector.select("random", k=k, seed=seed)

            for ts_key, test_set in test_sets.items():
                n_todo = sum(
                    1 for img, (_, row) in zip(test_set["images"], test_set["rows"].iterrows())
                    if (k, "random", seed, test_set["name"], str(row["image_id"])) not in done_keys
                    and img.exists()
                )
                if n_todo == 0:
                    print(f"  [{test_set['name']}] already done — skipping")
                    continue

                print(f"  [{test_set['name']}] {n_todo} images to process")
                records = run_on_test_set(
                    engine, examples, test_set,
                    k=k, strategy="random", seed=seed,
                    experiment_type="k_shot_ablation",
                    cfg=cfg, done_keys=done_keys,
                )
                all_records.extend(records)
                done_keys.update(
                    (r["k_shot"], r["strategy"], r["seed"], r["test_set"], r["image_id"])
                    for r in records
                )
                _append_jsonl(ckpt_path, records)
                print(f"    → {len(records)} records saved to {ckpt_path.name}")

    return all_records


# ── Experiment 2: Strategy Comparison ─────────────────────────────────────

def run_experiment2(engine, selector, test_sets, cfg, output_dir, done_keys, best_k=4):
    """Strategy comparison: {random,diversity,similarity} × seeds × test_sets at best_k."""
    exp_cfg   = cfg["experiments"]
    ckpt_path = output_dir / "checkpoint_exp2.jsonl"
    all_records = []

    print("\n" + "=" * 60)
    print(f"EXPERIMENT 2: Selection Strategy Comparison (k={best_k})")
    print("=" * 60)

    for strategy in exp_cfg["selection_strategies"]:
        for seed in range(exp_cfg["num_random_seeds"]):
            exp_id = f"strategy={strategy} k={best_k} seed={seed}"
            print(f"\n--- {exp_id} ---")

            # For similarity, pass RS UNAIR metadata as the target test distribution
            rs_meta = test_sets["rsunair_external"]["metadata"]
            examples = selector.select(strategy, k=best_k, seed=seed, test_metadata=rs_meta)

            for ts_key, test_set in test_sets.items():
                n_todo = sum(
                    1 for img, (_, row) in zip(test_set["images"], test_set["rows"].iterrows())
                    if (best_k, strategy, seed, test_set["name"], str(row["image_id"])) not in done_keys
                    and img.exists()
                )
                if n_todo == 0:
                    print(f"  [{test_set['name']}] already done — skipping")
                    continue

                print(f"  [{test_set['name']}] {n_todo} images to process")
                records = run_on_test_set(
                    engine, examples, test_set,
                    k=best_k, strategy=strategy, seed=seed,
                    experiment_type="selection_strategy",
                    cfg=cfg, done_keys=done_keys,
                )
                all_records.extend(records)
                done_keys.update(
                    (r["k_shot"], r["strategy"], r["seed"], r["test_set"], r["image_id"])
                    for r in records
                )
                _append_jsonl(ckpt_path, records)
                print(f"    → {len(records)} records saved to {ckpt_path.name}")

    return all_records


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/experiment_config.yaml")
    parser.add_argument("--exp",     type=int, default=0,
                        help="0=both, 1=Exp1 only, 2=Exp2 only")
    parser.add_argument("--best_k",  type=int, default=None,
                        help="Override best_k for Exp2 (default: read from results or 4)")
    args = parser.parse_args()

    # ── Config ─────────────────────────────────────────────────────────────
    cfg      = load_config(args.config)
    data_root = Path("data")
    output_dir = Path(cfg["experiments"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load test sets ──────────────────────────────────────────────────────
    test_sets = load_test_sets(cfg, data_root)

    # ── Load model ──────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    engine = NVReasonCXREngine(
        model_path=model_cfg["name"],
        quantize=model_cfg.get("quantize", "4bit"),
        min_pixels=model_cfg.get("min_pixels", 256),
        max_pixels=model_cfg.get("max_pixels", 1024),
    )

    # ── Init selector ──────────────────────────────────────────────────────
    selector = FewShotSelector(
        positive_csv="data/chexpert_ptx_positive.csv",
        negative_csv="data/chexpert_ptx_negative.csv",
    )

    # ── Resume: load already completed records ──────────────────────────────
    done_keys: set = set()
    for ckpt in output_dir.glob("checkpoint_exp*.jsonl"):
        done_keys |= _load_done_keys(ckpt)
    print(f"[Resume] {len(done_keys)} records already done")

    # ── Run experiments ─────────────────────────────────────────────────────
    if args.exp in (0, 1):
        run_experiment1(engine, selector, test_sets, cfg, output_dir, done_keys)

    if args.exp in (0, 2):
        # Determine best_k for Exp2
        best_k = args.best_k
        if best_k is None:
            # Try to infer from Exp1 results (pick k with highest mean accuracy)
            ckpt1 = output_dir / "checkpoint_exp1.jsonl"
            if ckpt1.exists():
                from src.evaluate import extract_diagnosis, compute_classification_metrics
                records = [json.loads(l) for l in open(ckpt1, encoding="utf-8")]
                df = pd.DataFrame(records)
                df["predicted"] = df["generated_text"].apply(extract_diagnosis)
                accs = (
                    df[df["test_set"] == "CheXpert Plus (Internal)"]
                    .groupby("k_shot")
                    .apply(lambda g: (g["predicted"] == g["ground_truth"]).mean())
                )
                best_k = int(accs.idxmax()) if len(accs) > 0 else 4
                print(f"[Exp2] Inferred best_k={best_k} from Exp1 results")
            else:
                best_k = 4
                print(f"[Exp2] No Exp1 results yet — using best_k={best_k}")

        run_experiment2(engine, selector, test_sets, cfg, output_dir, done_keys, best_k=best_k)

    # ── Merge all checkpoints into final results ────────────────────────────
    all_records = []
    for ckpt in sorted(output_dir.glob("checkpoint_exp*.jsonl")):
        with open(ckpt, encoding="utf-8") as f:
            for line in f:
                try:
                    all_records.append(json.loads(line))
                except Exception:
                    pass

    # Deduplicate
    seen = set()
    unique_records = []
    for r in all_records:
        key = (r["k_shot"], r["strategy"], r["seed"], r["test_set"], r["image_id"])
        if key not in seen:
            seen.add(key)
            unique_records.append(r)

    final_path = output_dir / "all_results.jsonl"
    with open(final_path, "w", encoding="utf-8") as f:
        for r in unique_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Experiments complete.")
    print(f"Total unique records: {len(unique_records)}")
    print(f"Final results: {final_path}")


if __name__ == "__main__":
    main()
