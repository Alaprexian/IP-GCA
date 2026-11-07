"""
Local Causal LLM Generator for IP-GCA Copilot.
Integrates Qwen2.5-1.5B-Instruct with 4-bit quantization (BitsAndBytes)
for low-latency, memory-efficient grounded text synthesis.
"""

from typing import List, Dict, Optional, Any, Union
import os
from pathlib import Path
import logging
import torch

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    Manages local causal language model inference for grounded generation and adaptation.
    """

    def __init__(
        self,
        model_name_or_path: str = "models/Qwen2.5-1.5B-Instruct",
        load_in_4bit: bool = True,
        device: Optional[str] = None,
        lazy_load: bool = False
    ):
        self.model_path = model_name_or_path
        self.load_in_4bit = load_in_4bit
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

        if not lazy_load:
            self.load()

    def load(self) -> None:
        """Loads tokenizer and weights into device memory."""
        if self._is_loaded:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        resolved_path = Path(self.model_path)
        if not resolved_path.exists():
            # If relative path from project root
            resolved_path = Path(__file__).resolve().parent.parent.parent / self.model_path

        model_identifier = str(resolved_path) if resolved_path.exists() else self.model_path

        logger.info(f"Cargando tokenizer desde: {model_identifier}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_identifier,
            trust_remote_code=True
        )

        if self.load_in_4bit and torch.cuda.is_available() and self.device_str != "cpu":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            logger.info("Cargando modelo Qwen con cuantización NF4 (4-bit)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            dtype = torch.float16 if (torch.cuda.is_available() and self.device_str != "cpu") else torch.float32
            logger.info(f"Cargando modelo en precisión: {dtype} en {self.device_str}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                torch_dtype=dtype,
                trust_remote_code=True
            )
            if self.device_str != "cpu" and torch.cuda.is_available():
                self.model = self.model.to(self.device_str)

        self.model.eval()
        self._is_loaded = True
        logger.info(f"Modelo cargado exitosamente en dispositivo: {self.model.device}")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def generate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05
    ) -> str:
        """
        Executes causal autoregressive generation using chat template formatting.
        """
        if not self._is_loaded:
            self.load()

        if messages is None:
            formatted_messages = []
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            formatted_messages.append({"role": "user", "content": prompt or ""})
        else:
            formatted_messages = messages

        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = "\n\n".join(m.get("content", "") for m in formatted_messages)

        inputs = self.tokenizer(input_text, return_tensors="pt")
        target_device = self.model.device
        input_ids = inputs["input_ids"].to(target_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_device)

        prompt_len = input_ids.shape[-1]
        do_sample = temperature > 0.01

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Slice off prompt tokens to decode only new generated response
        generated_tokens = output_ids[0][prompt_len:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return response_text
