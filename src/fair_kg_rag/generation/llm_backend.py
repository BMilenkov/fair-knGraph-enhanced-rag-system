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
        logger.info(f"Model loaded: {self.model_name}")

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

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
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
    ) -> list[str]:
        """Generate text for a batch of prompts.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Override default max tokens.

        Returns:
            List of generated texts.
        """
        # Simple sequential generation (batch padding is complex)
        return [self.generate(p, max_new_tokens=max_new_tokens) for p in prompts]

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
