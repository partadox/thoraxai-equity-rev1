"""
Inference engines for chest X-ray VLM experiments.

Two engines:
  - NVReasonCXREngine: Original engine for NV-Reason-CXR-3B (Qwen2.5-VL only).
    Used by Exp1 v1 notebook. Takes chat-formatted messages.

  - MultiModelEngine: Unified engine for 5 models across 3 backends.
    Used by Exp1/Exp2 v2 notebooks. Takes (system_prompt, user_text, image).
    Backends: "qwen" (Qwen2.5-VL family), "gemma" (MedGemma), "internvl" (InternVL2).

Usage:
    # Original (v1)
    engine = NVReasonCXREngine(model_path="./models/NV-Reason-CXR-3B", quantize="4bit")
    text, t = engine.generate(messages)

    # Multi-model (v2)
    engine = MultiModelEngine("./models/MedGemma-4B-IT", backend="gemma")
    text, t = engine.generate(system_prompt, user_text, image, max_new_tokens=1024)
"""

import gc
import time
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import BitsAndBytesConfig


def _extract_images(messages: list[dict]) -> list[Image.Image]:
    """Extract all PIL images from messages in order (examples + query)."""
    images = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        for item in msg.get("content", []):
            if item.get("type") == "image":
                img = item["image"]
                if isinstance(img, Image.Image):
                    images.append(img)
                elif isinstance(img, (str, Path)):
                    images.append(Image.open(str(img)).convert("RGB"))
    return images


class NVReasonCXREngine:
    """
    Inference engine for NV-Reason-CXR-3B.

    Args:
        model_path : local path or HuggingFace repo id
        quantize   : False | "8bit" | "4bit"
                     Use "4bit" for GPUs with ≤ 12 GB VRAM (RTX 5060 = 8.5 GB)
        min_pixels : min image resolution in Qwen2.5-VL patch units (default 256)
        max_pixels : max image resolution in Qwen2.5-VL patch units (default 1024)
    """

    def __init__(
        self,
        model_path: str = "./models/NV-Reason-CXR-3B",
        quantize: bool | str = "4bit",
        min_pixels: int = 256,
        max_pixels: int = 1024,
    ):
        print(f"[Engine] Loading model: {model_path}")
        print(f"[Engine] Quantize: {quantize}")

        # ── Processor ──────────────────────────────────────────────────────
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels * 28 * 28,
            max_pixels=max_pixels * 28 * 28,
        )

        # ── Model ──────────────────────────────────────────────────────────
        load_kwargs: dict = {
            "dtype"      : torch.float16,
            "device_map" : "auto",
        }

        if quantize == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )
        elif quantize == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, **load_kwargs
        )
        self.model.eval()

        vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"[Engine] Model loaded. VRAM used: {vram_gb:.2f} GB")

    # ── Single inference ───────────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> tuple[str, float]:
        """
        Run inference on a single set of messages.

        Args:
            messages       : chat messages from prompt_builder.build_messages()
            max_new_tokens : max tokens to generate
            temperature    : sampling temperature (low = more deterministic)
            top_p          : nucleus sampling threshold

        Returns:
            (generated_text, inference_time_seconds)
        """
        # Apply chat template → text string (with image placeholders)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Extract all images (few-shot examples + query) in order
        images = _extract_images(messages)

        # Tokenize + process images
        inputs = self.processor(
            text=text,
            images=images if images else None,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        do_sample = temperature > 0
        gen_kwargs: dict = {
            "max_new_tokens" : max_new_tokens,
            "do_sample"      : do_sample,
            "pad_token_id"   : self.processor.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"]       = top_p

        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        inference_time = time.time() - t0

        # Decode only generated (new) tokens
        input_len      = inputs["input_ids"].shape[1]
        generated_ids  = output_ids[:, input_len:]
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Free GPU memory
        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generated_text, inference_time

    # ── Batch inference ────────────────────────────────────────────────────

    def batch_inference(
        self,
        messages_list: list[list[dict]],
        progress_prefix: str = "",
        **generate_kwargs,
    ) -> list[dict]:
        """
        Run inference sequentially on a list of message sets.

        Returns:
            list of {"text": str, "inference_time": float, "error": str|None}
        """
        results = []
        n = len(messages_list)
        for i, messages in enumerate(messages_list):
            tag = f"{progress_prefix}[{i+1}/{n}]"
            print(f"  {tag} generating...", end=" ", flush=True)
            try:
                text, t = self.generate(messages, **generate_kwargs)
                print(f"{t:.1f}s")
                results.append({"text": text, "inference_time": t, "error": None})
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                print("OOM — skipped")
                results.append({"text": "", "inference_time": 0.0, "error": "OOM"})
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"text": "", "inference_time": 0.0, "error": str(e)})
        return results

    def vram_used_gb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0


# ══════════════════════════════════════════════════════════════════════════
# MultiModelEngine — unified inference for 5 models, 3 backends
# ══════════════════════════════════════════════════════════════════════════

_QUANT_CONFIG_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


class MultiModelEngine:
    """
    Unified inference engine supporting multiple VLM backends.

    Backends:
      - "qwen"     : Qwen2.5-VL family (NV-Reason-CXR, ChestXReasoner, Qwen2.5-VL)
      - "gemma"    : MedGemma (Gemma 3)
      - "internvl" : Mini-InternVL2

    Args:
        model_path    : local path or HuggingFace repo id
        backend       : "qwen", "gemma", or "internvl"
        quantize_4bit : use 4-bit quantization (default True)
        user_query    : text appended after image in user turn (default "Analyze this chest X-ray.")
        min_pixels    : Qwen2.5-VL min image patches (default 256)
        max_pixels    : Qwen2.5-VL max image patches (default 1024)
    """

    def __init__(
        self,
        model_path: str,
        backend: str,
        quantize_4bit: bool = True,
        user_query: str = "Analyze this chest X-ray.",
        min_pixels: int = 256,
        max_pixels: int = 1024,
    ):
        self.backend = backend
        self.model_path = model_path
        self.user_query = user_query
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels

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
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=self._min_pixels * 28 * 28,
            max_pixels=self._max_pixels * 28 * 28,
        )
        load_kw = {"dtype": torch.float16, "device_map": "auto"}
        if quantize_4bit:
            load_kw["quantization_config"] = _QUANT_CONFIG_4BIT
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kw)

    def _generate_qwen(self, system_prompt, user_text, image, **kwargs):
        content_parts = []
        if user_text:
            content_parts.append({"type": "text", "text": user_text})
        content_parts.append({"type": "image", "image": image})
        content_parts.append({"type": "text", "text": self.user_query})

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
        gen_kw["pad_token_id"] = self.processor.tokenizer.eos_token_id

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kw)
        generated = out[:, inputs["input_ids"].shape[1]:]
        result = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        del inputs, out, generated
        return result

    # ── Backend: Gemma 3 (MedGemma) ────────────────────────────────────

    def _load_gemma(self, model_path, quantize_4bit):
        from transformers import Gemma3ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path)
        load_kw = {"dtype": torch.float16, "device_map": "auto"}
        if quantize_4bit:
            load_kw["quantization_config"] = _QUANT_CONFIG_4BIT
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_path, **load_kw)

    def _generate_gemma(self, system_prompt, user_text, image, **kwargs):
        content_parts = []
        if user_text:
            content_parts.append({"type": "text", "text": user_text})
        content_parts.append({"type": "image", "image": image})
        content_parts.append({"type": "text", "text": self.user_query})

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
            load_kw["quantization_config"] = _QUANT_CONFIG_4BIT
        self.model = AutoModel.from_pretrained(model_path, **load_kw)
        self.processor = None

    def _generate_internvl(self, system_prompt, user_text, image, **kwargs):
        full_prompt = f"{system_prompt}\n\n"
        if user_text:
            full_prompt += f"{user_text}\n\n"
        full_prompt += f"<image>\n{self.user_query}"

        gen_kw = self._gen_kwargs(**kwargs)

        if hasattr(self.model, "chat"):
            result = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=self._internvl_preprocess_image(image),
                question=full_prompt,
                generation_config=gen_kw,
            )
        else:
            result = self._internvl_manual_generate(full_prompt, image, **gen_kw)
        return result

    def _internvl_preprocess_image(self, image):
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(
            device=self.model.device, dtype=torch.float16
        )

    def _internvl_manual_generate(self, prompt, image, **gen_kw):
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
