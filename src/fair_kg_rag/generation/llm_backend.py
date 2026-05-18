"""HuggingFace local LLM interface for generation and extraction."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class LLMBackend:
    """Unified local LLM interface using HuggingFace transformers.

    Supports loading models with optional 4-bit quantization for
    memory-efficient inference on consumer GPUs.

    Args:
        model_name: HuggingFace model identifier.
        device: Device for inference ("cuda" or "cpu").
        max_new_tokens: Default maximum tokens to generate.
        temperature: Sampling temperature.
        use_4bit: Whether to use 4-bit quantization.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = "cuda",
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        use_4bit: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_4bit = use_4bit
        self._model = None
        self._tokenizer = None
        self._use_chat_template = False

    def _load_model(self) -> None:
        """Lazy-load the model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}

        # Use SDPA (Scaled Dot-Product Attention) for faster attention on T4
        # Note: Flash Attention 2 requires SM80+ (Ampere); T4 is SM75 (Turing)
        if self.device == "cuda":
            load_kwargs["attn_implementation"] = "sdpa"

        if self.use_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )
        self._model.eval()

        # Use chat template if the tokenizer provides one (e.g. Mistral [INST])
        if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
            self._use_chat_template = True
            logger.info("Chat template detected — will wrap prompts with [INST] format")

        logger.info(f"Model loaded: {self.model_name}")

    def _format_prompt(self, prompt: str) -> str:
        """Wrap prompt with chat template (e.g. Mistral [INST]) if available."""
        if not self._use_chat_template:
            return prompt
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool = False,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            do_sample: Whether to use sampling.

        Returns:
            Generated text (excluding the prompt).
        """
        self._load_model()

        max_tokens = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature

        formatted = self._format_prompt(prompt)
        inputs = self._tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=4096,
            add_special_tokens=not self._use_chat_template,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the generated portion
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int | None = None,
        do_sample: bool = False,
        max_batch_tokens: int = 16384,
        max_batch_size: int = 32,
    ) -> list[str]:
        """Generate text using sorted adaptive micro-batching.

        Sorts prompts by token length and creates variable-size micro-batches
        constrained by a token budget. Short prompts get large batches (more
        GPU parallelism), long prompts get small batches (avoid OOM). Padding
        waste is minimized because similar-length prompts are grouped together.

        Args:
            prompts: List of input prompts (any number).
            max_new_tokens: Override default max tokens.
            do_sample: Whether to use sampling.
            max_batch_tokens: Token budget per micro-batch (input + output).
            max_batch_size: Hard cap on items per micro-batch.

        Returns:
            List of generated texts (same order as input prompts).
        """
        if not prompts:
            return []

        self._load_model()
        max_tokens = max_new_tokens or self.max_new_tokens
        add_special = not self._use_chat_template

        # Format all prompts with chat template
        formatted = [self._format_prompt(p) for p in prompts]

        # Pre-tokenize for exact lengths (fast: ~1ms per prompt)
        token_lengths = [
            len(self._tokenizer.encode(f, add_special_tokens=add_special))
            for f in formatted
        ]

        # Sort by token length → minimal padding within each micro-batch
        order = sorted(range(len(formatted)), key=lambda i: token_lengths[i])
        formatted_sorted = [formatted[i] for i in order]
        lengths_sorted = [token_lengths[i] for i in order]

        if len(prompts) > 1:
            logger.info(
                f"Smart batching {len(prompts)} prompts "
                f"(tokens: {lengths_sorted[0]}–{lengths_sorted[-1]}, "
                f"budget: {max_batch_tokens})"
            )

        results_sorted: list[str] = []
        prev_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"

        pos = 0
        while pos < len(formatted_sorted):
            # Binary search for optimal micro-batch size:
            # find largest bs where longest_seq * bs <= token budget
            lo, hi = 1, min(max_batch_size, len(formatted_sorted) - pos)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                seq_len = lengths_sorted[pos + mid - 1] + max_tokens
                if seq_len * mid <= max_batch_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            bs = lo

            batch = formatted_sorted[pos:pos + bs]

            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
                add_special_tokens=add_special,
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for output in outputs:
                generated_ids = output[input_len:]
                text = self._tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                results_sorted.append(text)

            pos += bs

        self._tokenizer.padding_side = prev_side

        # Restore original order
        results = [""] * len(prompts)
        for sorted_idx, orig_idx in enumerate(order):
            results[orig_idx] = results_sorted[sorted_idx]

        return results

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
