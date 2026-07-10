#!/usr/bin/env python3
"""Stage-1 k-shot inference runner (M1/M3/M4) on corrected data2.
Faithful port of experiment-1-v2-data2/M1.ipynb, parametrized by model,
with Gemma->bfloat16 fix and output under final_rerun_paper_revision/.
Usage: python -u run_stage1.py M1|M3|M4 [--sanity]
"""
import argparse
ARGS = argparse.ArgumentParser()
ARGS.add_argument("model", choices=["M1","M3","M4"])
ARGS.add_argument("--sanity", action="store_true")
ARGS = ARGS.parse_args()


# ===== c2 =====
import sys
import json
import yaml
import re
import time
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import os

import torch

PROJECT_ROOT = Path("/data/arta/thoraxai_research/1_equity")
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

print(f"Working dir : {os.getcwd()}")
print(f"Python      : {sys.executable}")
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA        : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ===== c4 =====
# ╔══════════════════════════════════════════════════════════════╗
# ║  GANTI DI SINI — pilih salah satu: "M1", "M2", ... "M5"   ║
# ╚══════════════════════════════════════════════════════════════╝
ACTIVE_MODEL = ARGS.model


MODEL_REGISTRY = {
    "M1": {
        "name":       "NV-Reason-CXR-3B",
        "path":       "./models/NV-Reason-CXR-3B",
        "backend":    "qwen",
        "short_id":   "nv_reason",
    },
    "M2": {
        "name":       "ChestXReasoner",
        "path":       "./models/ChestXReasoner",
        "backend":    "qwen",
        "short_id":   "chestxreasoner",
    },
    "M3": {
        "name":       "Qwen2.5-VL-3B-Instruct",
        "path":       "./models/Qwen2.5-VL-3B-Instruct",
        "backend":    "qwen",
        "short_id":   "qwen25vl_base",
    },
    "M4": {
        "name":       "MedGemma-4B-IT",
        "path":       "./models/MedGemma-4B-IT",
        "backend":    "gemma",
        "short_id":   "medgemma",
    },
    "M5": {
        "name":       "Mini-InternVL2-4B-Medical",
        "path":       "./models/Mini-InternVL2-4B-Medical",
        "backend":    "internvl",
        "short_id":   "internvl_med",
    },
}

M = MODEL_REGISTRY[ACTIVE_MODEL]
MODEL_NAME    = M["name"]
MODEL_PATH    = M["path"]
MODEL_BACKEND = M["backend"]
MODEL_SHORT   = M["short_id"]

print(f"\n{'='*60}")
print(f"  ACTIVE MODEL: [{ACTIVE_MODEL}] {MODEL_NAME}")
print(f"  Backend     : {MODEL_BACKEND}")
print(f"  Path        : {MODEL_PATH}")
print(f"{'='*60}")


if ACTIVE_MODEL not in ("M1","M3","M4"):
    raise SystemExit(f"This rerun supports M1/M3/M4 only (got {ACTIVE_MODEL}). M2 needs gated access; M5 not used.")

# ===== c6 =====
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_config2.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

DATA_ROOT  = PROJECT_ROOT / "data2"
RERUN_ROOT = PROJECT_ROOT / "final_rerun_paper_revision"
OUTPUT_DIR = RERUN_ROOT / ("results_sanity" if ARGS.sanity else "results") / f"exp1v2_{MODEL_SHORT}"
CKPT_PATH  = OUTPUT_DIR / "checkpoint.jsonl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_SHOTS    = cfg["experiments"]["k_shots"]          # [0, 1, 2, 4, 8]
NUM_SEEDS  = cfg["experiments"]["num_random_seeds"]  # 10 (was 5) — Wilcoxon p-min = 0.00195
SEEDS      = list(range(NUM_SEEDS))

# Generation parameters (kontrol — sama untuk semua model)
GEN_PARAMS = {
    "max_new_tokens": cfg["model"]["max_new_tokens"],  # 1024
    "temperature":    cfg["model"]["temperature"],      # 0.1
    "top_p":          cfg["model"]["top_p"],            # 0.9
}

print(f"k_shots     : {K_SHOTS}")
print(f"seeds       : {SEEDS}")
print(f"output_dir  : {OUTPUT_DIR}")
print(f"checkpoint  : {CKPT_PATH}")
print(f"gen_params  : {GEN_PARAMS}")


if ARGS.sanity:
    K_SHOTS = [0]
    SEEDS = [0]
    print(">>> SANITY MODE: k=[0], seeds=[0], max 3 images/test-set")

# ===== c8 =====
# ╔══════════════════════════════════════════════════════════════╗
# ║  PROMPT V2 — Balanced, tidak bias ke pneumothorax          ║
# ║  Perubahan dari v1:                                        ║
# ║  - Dihapus "particular expertise in pneumothorax"          ║
# ║  - Ditambah "A normal CXR should be classified as NO PTX" ║
# ║  - User query: "Analyze this chest X-ray." (no "for PTX") ║
# ║  - Few-shot: "Evaluate independently" ditambahkan         ║
# ╚══════════════════════════════════════════════════════════════╝

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

_BOILERPLATE_PAT = re.compile(
    r"I have personally reviewed\b.*?(?:\.\s|$)",
    re.IGNORECASE | re.DOTALL,
)
_MAX_REPORT_WORDS = 60


def _clean_report(text: str) -> str:
    text = _BOILERPLATE_PAT.sub("", text)
    text = re.sub(r"[\x00-\x1f\r\n]+", " ", text).strip()
    words = text.split()
    if len(words) > _MAX_REPORT_WORDS:
        text = " ".join(words[:_MAX_REPORT_WORDS]) + "..."
    return text


def build_few_shot_context(examples: list[dict], k_shot: int) -> str:
    """Build text-based few-shot context block (prompt v2)."""
    positives = [ex for ex in examples if ex["category"] == "positive"]
    negatives = [ex for ex in examples if ex["category"] == "negative"]
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


print("Prompt v2 loaded.")
print(f"System prompt preview: {SYSTEM_PROMPT_V2[:80]}...")
print(f"User query: {USER_QUERY_V2}")


# ===== c10 =====
from transformers import BitsAndBytesConfig

QUANT_CONFIG_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=(torch.bfloat16 if MODEL_BACKEND == "gemma" else torch.float16),
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def _load_image(path) -> Image.Image:
    img = Image.open(str(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


class MultiModelEngine:
    """
    Unified inference engine supporting multiple VLM backends.

    Backends:
      - "qwen"     : Qwen2.5-VL family (NV-Reason-CXR, ChestXReasoner, Qwen2.5-VL)
      - "gemma"    : MedGemma (Gemma 3)
      - "internvl" : Mini-InternVL2
    """

    def __init__(self, model_path: str, backend: str, quantize_4bit: bool = True):
        self.backend = backend
        self.model_path = model_path
        print(f"[Engine] Loading: {model_path} (backend={backend}, 4bit={quantize_4bit})")

        if backend == "qwen":
            self._load_qwen(model_path, quantize_4bit)
        elif backend == "gemma":
            self._load_gemma(model_path, quantize_4bit)
        elif backend == "internvl":
            self._load_internvl(model_path, quantize_4bit)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.model.eval()
        print(f"[Engine] Ready. VRAM: {self.vram_used_gb():.2f} GB")

    # ── Backend: Qwen2.5-VL ────────────────────────────────────────────

    def _load_qwen(self, model_path, quantize_4bit):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        min_px = cfg["model"].get("min_pixels", 256) * 28 * 28
        max_px = cfg["model"].get("max_pixels", 1024) * 28 * 28

        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=min_px, max_pixels=max_px,
        )
        load_kw = {"dtype": torch.float16, "device_map": "auto"}
        if quantize_4bit:
            load_kw["quantization_config"] = QUANT_CONFIG_4BIT
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kw)

    def _generate_qwen(self, system_prompt, user_text, image, **kwargs):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_text} if user_text else None,
                {"type": "image", "image": image},
                {"type": "text", "text": USER_QUERY_V2},
            ]},
        ]
        # Remove None entries
        messages[1]["content"] = [c for c in messages[1]["content"] if c is not None]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text, images=[image], return_tensors="pt",
        ).to(self.model.device)

        gen_kw = self._gen_kwargs(**kwargs)
        gen_kw["pad_token_id"] = self.processor.tokenizer.eos_token_id

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kw)

        generated = out[:, inputs["input_ids"].shape[1]:]
        result = self.processor.batch_decode(generated, skip_special_tokens=True)[0]

        del inputs, out, generated
        return result

    # ── Backend: Gemma 3 (MedGemma) ────────────────────────────────────

    def _load_gemma(self, model_path, quantize_4bit):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path)
        load_kw = {"dtype": torch.bfloat16, "device_map": "auto"}  # Gemma3 needs bf16 (fp16 -> all-zero)
        if quantize_4bit:
            load_kw["quantization_config"] = QUANT_CONFIG_4BIT
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_path, **load_kw)

    def _generate_gemma(self, system_prompt, user_text, image, **kwargs):
        content_parts = []
        if user_text:
            content_parts.append({"type": "text", "text": user_text})
        content_parts.append({"type": "image", "image": image})
        content_parts.append({"type": "text", "text": USER_QUERY_V2})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content_parts},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text, images=[image], return_tensors="pt",
        ).to(self.model.device)

        gen_kw = self._gen_kwargs(**kwargs)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kw)

        generated = out[:, inputs["input_ids"].shape[1]:]
        result = self.processor.batch_decode(generated, skip_special_tokens=True)[0]

        del inputs, out, generated
        return result

    # ── Backend: InternVL ──────────────────────────────────────────────

    def _load_internvl(self, model_path, quantize_4bit):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        load_kw = {"dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}
        if quantize_4bit:
            load_kw["quantization_config"] = QUANT_CONFIG_4BIT
        self.model = AutoModel.from_pretrained(model_path, **load_kw)
        self.processor = None  # InternVL uses tokenizer + custom image processing

    def _generate_internvl(self, system_prompt, user_text, image, **kwargs):
        # InternVL uses <image> tag in prompt
        full_prompt = f"{system_prompt}\n\n"
        if user_text:
            full_prompt += f"{user_text}\n\n"
        full_prompt += f"<image>\n{USER_QUERY_V2}"

        gen_kw = self._gen_kwargs(**kwargs)

        # InternVL has its own chat method
        if hasattr(self.model, "chat"):
            result = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=self._internvl_preprocess_image(image),
                question=full_prompt,
                generation_config=gen_kw,
            )
        else:
            # Fallback: manual generate
            result = self._internvl_manual_generate(full_prompt, image, **gen_kw)

        return result

    def _internvl_preprocess_image(self, image):
        """Preprocess image for InternVL (resize + normalize + to tensor)."""
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = transform(image).unsqueeze(0).to(
            device=self.model.device, dtype=torch.float16
        )
        return pixel_values

    def _internvl_manual_generate(self, prompt, image, **gen_kw):
        """Fallback generate for InternVL without .chat() method."""
        pixel_values = self._internvl_preprocess_image(image)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, pixel_values=pixel_values, **gen_kw
            )
        generated = out[:, input_ids.shape[1]:]
        result = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        del pixel_values, input_ids, out, generated
        return result

    # ── Shared ─────────────────────────────────────────────────────────

    def _gen_kwargs(self, **kwargs):
        do_sample = kwargs.get("temperature", 0.1) > 0
        gen_kw = {
            "max_new_tokens": kwargs.get("max_new_tokens", 1024),
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kw["temperature"] = kwargs.get("temperature", 0.1)
            gen_kw["top_p"] = kwargs.get("top_p", 0.9)
        return gen_kw

    def generate(self, system_prompt, user_text, image, **kwargs):
        """
        Unified generate — same interface for all backends.

        Args:
            system_prompt : str — system message
            user_text     : str — few-shot context (empty string for 0-shot)
            image         : PIL.Image — the query CXR image
            **kwargs      : max_new_tokens, temperature, top_p

        Returns:
            (generated_text: str, inference_time: float)
        """
        t0 = time.time()

        if self.backend == "qwen":
            result = self._generate_qwen(system_prompt, user_text, image, **kwargs)
        elif self.backend == "gemma":
            result = self._generate_gemma(system_prompt, user_text, image, **kwargs)
        elif self.backend == "internvl":
            result = self._generate_internvl(system_prompt, user_text, image, **kwargs)

        elapsed = time.time() - t0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result, elapsed

    def vram_used_gb(self) -> float:
        return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


print("MultiModelEngine class defined.")


# ===== c12 =====
def _img_path_chexpert_test(file_id: str, label_str: str) -> Path:
    subfolder = "internal_test_pneumothorax" if label_str == "pneumothorax" else "internal_test_normal"
    return DATA_ROOT / "chexpert_plus" / "images" / subfolder / f"{file_id}.png"

def _img_path_rsunair(image_id: str, label_int: int) -> Path:
    subfolder = "pneumothorax" if label_int == 1 else "normal"
    return DATA_ROOT / "rs_unair" / "images" / subfolder / f"{image_id}.png"


# ── CheXpert Plus Internal Test ────────────────────────────────────────
chex_df = pd.read_csv(cfg["data"]["chexpert_internal_test_csv"], encoding="utf-8-sig")
chex_df["label_int"] = chex_df["label"].map({"pneumothorax": 1, "normal": 0})
chex_df["image_id"]  = chex_df["file_id"]
chex_images = [
    _img_path_chexpert_test(row["file_id"], row["label"])
    for _, row in chex_df.iterrows()
]
missing_chex = sum(1 for p in chex_images if not p.exists())
print(f"CheXpert Internal : {len(chex_df)} images  |  missing: {missing_chex}")
print(f"  PTX={chex_df['label_int'].sum()}  Normal={len(chex_df)-chex_df['label_int'].sum()}")


# ── RS UNAIR External ──────────────────────────────────────────────────
rs_labels = pd.read_csv(cfg["data"]["rsunair_labels_csv"], encoding="utf-8-sig")
rs_meta   = pd.read_csv(
    DATA_ROOT / "rs_unair" / "metadata_normalized_dicom.csv",
    encoding="utf-8-sig", sep=None, engine="python"
)
rs_labels_renamed  = rs_labels.rename(columns={"image_id": "file_id", "label": "label_int"})
rs_meta_merged     = rs_meta.merge(rs_labels_renamed, on="file_id", how="inner")
rs_meta_merged["image_id"] = rs_meta_merged["file_id"]

rs_images = [
    _img_path_rsunair(row["image_id"], row["label_int"])
    for _, row in rs_meta_merged.iterrows()
]
missing_rs = sum(1 for p in rs_images if not p.exists())
print(f"RS UNAIR External : {len(rs_meta_merged)} images  |  missing: {missing_rs}")
print(f"  PTX={rs_meta_merged['label_int'].sum()}  Normal={len(rs_meta_merged)-rs_meta_merged['label_int'].sum()}")


TEST_SETS = {
    "chexpert_internal": {
        "name"   : "CheXpert Plus (Internal)",
        "rows"   : chex_df.reset_index(drop=True),
        "images" : chex_images,
    },
    "rsunair_external": {
        "name"   : "RS UNAIR (External)",
        "rows"   : rs_meta_merged.reset_index(drop=True),
        "images" : rs_images,
    },
}

print("\nTest sets loaded.")


# ===== c14 =====
engine = MultiModelEngine(
    model_path=MODEL_PATH,
    backend=MODEL_BACKEND,
    quantize_4bit=True,
)

print(f"\nModel [{ACTIVE_MODEL}] {MODEL_NAME} siap.")
print(f"VRAM used: {engine.vram_used_gb():.2f} GB")


# ----- single-inference sanity (non-fatal) -----
try:
    _ti = _load_image(chex_images[0])
    _to, _tt = engine.generate(system_prompt=SYSTEM_PROMPT_V2, user_text="", image=_ti, **GEN_PARAMS)
    print(f"[sanity] {chex_images[0].name}: {_tt:.1f}s, {len(_to)} chars | preview: {_to[:160]!r}")
    _ti.close()
except Exception as _e:
    print(f"[sanity] WARNING: {_e}")

# ===== c18 =====
from src.example_selector import FewShotSelector

selector = FewShotSelector(
    positive_csv="data/chexpert_ptx_positive.csv",
    negative_csv="data/chexpert_ptx_negative.csv",
)


# ===== c20 =====
def load_done_keys(ckpt_path: Path) -> set:
    done = set()
    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["k_shot"], r["strategy"], r["seed"],
                              r["test_set"], r["image_id"]))
                except Exception:
                    pass
    return done


def append_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


done_keys = load_done_keys(CKPT_PATH)
print(f"Records sudah selesai (resume): {len(done_keys)}")

total_target = len(K_SHOTS) * len(SEEDS) * sum(len(ts["images"]) for ts in TEST_SETS.values())
print(f"Total target inferences        : {total_target}")
print(f"Sisa yang harus dijalankan     : {total_target - len(done_keys)}")


# ===== c22 =====
print("=" * 65)
print(f"EXPERIMENT 1 v2: K-Shot Ablation")
print(f"Model: [{ACTIVE_MODEL}] {MODEL_NAME}")
print(f"Prompt: v2 (balanced)")
print("=" * 65)

exp_start = datetime.now()

for k in K_SHOTS:
    for seed in SEEDS:
        # ── Select few-shot examples ────────────────────────────────
        examples = selector.select("random", k=k, seed=seed)

        # ── Build few-shot context text ─────────────────────────────
        fewshot_text = build_few_shot_context(examples, k_shot=k)

        # ── Run on BOTH test sets ───────────────────────────────────
        for ts_key, test_set in TEST_SETS.items():
            ts_name = test_set["name"]

            # Filter belum selesai
            todo = [
                (img_path, row)
                for img_path, (_, row) in zip(test_set["images"], test_set["rows"].iterrows())
                if (k, "random", seed, ts_name, str(row["image_id"])) not in done_keys
                and img_path.exists()
            ]

            if ARGS.sanity:
                todo = todo[:3]
            print(f"\n[k={k} seed={seed}] {ts_name}")

            if not todo:
                print("  \u2192 sudah selesai, skip.")
                continue

            print(f"  \u2192 {len(todo)} images to process...")
            batch_records = []

            for idx, (img_path, row) in enumerate(todo, 1):
                image_id = str(row["image_id"])
                print(f"    [{idx}/{len(todo)}] {image_id}", end=" ", flush=True)

                # Load image
                img = _load_image(img_path)

                try:
                    generated_text, t = engine.generate(
                        system_prompt=SYSTEM_PROMPT_V2,
                        user_text=fewshot_text,
                        image=img,
                        **GEN_PARAMS,
                    )
                    print(f"({t:.1f}s) \u2713")
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    generated_text, t = "", 0.0
                    print("OOM \u2717")
                except Exception as e:
                    generated_text, t = "", 0.0
                    print(f"ERROR: {e} \u2717")

                record = {
                    "experiment_type" : "k_shot_ablation",
                    "model_id"        : ACTIVE_MODEL,
                    "model_name"      : MODEL_NAME,
                    "prompt_version"  : "v2",
                    "k_shot"          : k,
                    "strategy"        : "random",
                    "seed"            : seed,
                    "test_set"        : ts_name,
                    "image_id"        : image_id,
                    "ground_truth"    : int(row["label_int"]),
                    "generated_text"  : generated_text,
                    "inference_time"  : round(t, 3),
                    "timestamp"       : datetime.now().isoformat(),
                }
                batch_records.append(record)
                done_keys.add((k, "random", seed, ts_name, image_id))

                # Flush setiap 10 records
                if len(batch_records) % 10 == 0:
                    append_jsonl(CKPT_PATH, batch_records)
                    batch_records = []

                # Close image to free memory
                img.close()

            # Flush sisa
            if batch_records:
                append_jsonl(CKPT_PATH, batch_records)

            print(f"  \u2192 checkpoint saved.")

elapsed = (datetime.now() - exp_start).total_seconds() / 3600
print(f"\n{'=' * 65}")
print(f"[{ACTIVE_MODEL}] {MODEL_NAME} — Experiment 1 v2 selesai.")
print(f"Total waktu sesi ini : {elapsed:.2f} jam")
print(f"Total records        : {len(done_keys)}")


# ===== c24 =====
from src.evaluate import extract_diagnosis, compute_classification_metrics

# Load checkpoint
records = []
if CKPT_PATH.exists():
    with open(CKPT_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                pass

print(f"Total records: {len(records)}")

df = pd.DataFrame(records)
df["predicted"] = df["generated_text"].apply(extract_diagnosis)

print(f"Invalid predictions: {(df['predicted'] == -1).sum()}")
print(f"k_shots: {sorted(df['k_shot'].unique())}")
print(f"test_sets: {df['test_set'].unique().tolist()}")


# ===== c25 =====
# Quick summary: mean ± std across seeds
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

rows = []
for (k, ts), grp in df.groupby(["k_shot", "test_set"]):
    seed_metrics = []
    for seed in grp["seed"].unique():
        sg = grp[grp["seed"] == seed]
        m = compute_classification_metrics(sg["ground_truth"].tolist(), sg["predicted"].tolist())
        seed_metrics.append(m)
    sm_df = pd.DataFrame(seed_metrics)
    rows.append({
        "k_shot": k, "test_set": ts,
        "acc":  f"{sm_df['accuracy'].mean():.4f}\u00b1{sm_df['accuracy'].std():.4f}",
        "sens": f"{sm_df['sensitivity'].mean():.4f}\u00b1{sm_df['sensitivity'].std():.4f}",
        "spec": f"{sm_df['specificity'].mean():.4f}\u00b1{sm_df['specificity'].std():.4f}",
        "f1":   f"{sm_df['f1'].mean():.4f}\u00b1{sm_df['f1'].std():.4f}",
    })

result_df = pd.DataFrame(rows)

print(f"\n{'='*70}")
print(f"  [{ACTIVE_MODEL}] {MODEL_NAME} — Quick Summary (Prompt v2)")
print(f"{'='*70}")

for ts in result_df["test_set"].unique():
    print(f"\n  {ts}:")
    sub = result_df[result_df["test_set"] == ts]
    for _, r in sub.iterrows():
        print(f"    k={int(r['k_shot']):d}  Acc={r['acc']}  F1={r['f1']}  Sens={r['sens']}  Spec={r['spec']}")


# ===== c27 =====
seen = set()
unique_records = []
for r in records:
    key = (r["k_shot"], r["strategy"], r["seed"], r["test_set"], r["image_id"])
    if key not in seen:
        seen.add(key)
        unique_records.append(r)

final_path = OUTPUT_DIR / "all_results.jsonl"
with open(final_path, "w", encoding="utf-8") as f:
    for r in unique_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Saved: {final_path}")
print(f"Records: {len(unique_records)} unique")


# ===== c29 =====
print(f"\n{'='*70}")
print(f"  BIAS ANALYSIS — [{ACTIVE_MODEL}] {MODEL_NAME}")
print(f"{'='*70}")

for ts in df["test_set"].unique():
    ts_df = df[df["test_set"] == ts]
    print(f"\n  {ts}:")
    print(f"  {'k':>3} | {'PBR':>6} | {'Spec':>6} | {'SDR':>7} | {'Sens':>6}")
    print(f"  {'-'*40}")

    spec_k0 = None

    for k in sorted(ts_df["k_shot"].unique()):
        k_df = ts_df[ts_df["k_shot"] == k]
        n_actual_pos = (k_df["ground_truth"] == 1).sum()
        n_pred_pos   = (k_df["predicted"] == 1).sum()

        pbr = n_pred_pos / n_actual_pos if n_actual_pos > 0 else 0

        tp = ((k_df["predicted"] == 1) & (k_df["ground_truth"] == 1)).sum()
        tn = ((k_df["predicted"] == 0) & (k_df["ground_truth"] == 0)).sum()
        fp = ((k_df["predicted"] == 1) & (k_df["ground_truth"] == 0)).sum()
        fn = ((k_df["predicted"] == 0) & (k_df["ground_truth"] == 1)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        if k == 0:
            spec_k0 = spec
            sdr = 0.0
        else:
            sdr = ((spec_k0 - spec) / spec_k0 * 100) if spec_k0 and spec_k0 > 0 else 0

        print(f"  {k:3d} | {pbr:6.2f} | {spec:6.3f} | {sdr:6.1f}% | {sens:6.3f}")


# ===== c31 =====
# Status check: model mana saja yang sudah ada hasilnya?
results_root = PROJECT_ROOT / "results"

print("\nStatus semua model:")
for mid, info in MODEL_REGISTRY.items():
    result_dir = results_root / f"exp1v2_{info['short_id']}"
    result_file = result_dir / "all_results.jsonl"
    if result_file.exists():
        n_lines = sum(1 for _ in open(result_file))
        status = f"DONE ({n_lines} records)"
    elif (result_dir / "checkpoint.jsonl").exists():
        n_lines = sum(1 for _ in open(result_dir / "checkpoint.jsonl"))
        status = f"IN PROGRESS ({n_lines} records)"
    else:
        status = "NOT STARTED"

    marker = "\u2714" if "DONE" in status else "\u25cb" if "PROGRESS" in status else "\u2717"
    current = " \u2190 CURRENT" if mid == ACTIVE_MODEL else ""
    print(f"  {marker} [{mid}] {info['name']:30s} {status}{current}")

