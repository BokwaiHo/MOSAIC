"""
LLM inference backend – thin wrapper around transformers / vLLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    tokens_used: int
    logprobs: Optional[torch.Tensor] = None   # (seq_len, vocab) or None
    hidden_states: Optional[torch.Tensor] = None  # last-layer hidden for scheduler


class LLMBackend:
    """Unified interface for backbone LLM inference."""

    def __init__(
        self,
        model_name: str,
        backend: str = "transformers",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.backend = backend
        self._model = None
        self._tokenizer = None
        self._device = device
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self):
        if self.backend == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_transformers(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading {self.model_name} via transformers …")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
            trust_remote_code=True,
        )
        self._model.eval()

    def _load_vllm(self):
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vllm is required.  pip install vllm")
        logger.info(f"Loading {self.model_name} via vLLM …")
        self._model = LLM(
            model=self.model_name,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        return_hidden: bool = False,
        return_logprobs: bool = False,
    ) -> LLMResponse:
        if self.backend == "vllm":
            return self._generate_vllm(
                prompt, max_new_tokens, temperature,
                return_hidden, return_logprobs,
            )
        return self._generate_transformers(
            prompt, max_new_tokens, temperature,
            return_hidden, return_logprobs,
        )

    def _generate_transformers(
        self, prompt, max_new_tokens, temperature,
        return_hidden, return_logprobs,
    ) -> LLMResponse:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs.input_ids.shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            output_hidden_states=return_hidden,
            output_scores=return_logprobs,
            return_dict_in_generate=True,
        )

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        new_ids = outputs.sequences[0, input_len:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        tokens_used = len(new_ids)

        hidden = None
        if return_hidden and hasattr(outputs, "hidden_states") and outputs.hidden_states:
            # outputs.hidden_states is tuple of (step, layers, batch, seq, dim)
            # Take last layer, last generated token
            last_step = outputs.hidden_states[-1]
            hidden = last_step[-1][0, -1, :]  # (dim,)

        logprobs = None
        if return_logprobs and hasattr(outputs, "scores") and outputs.scores:
            logprobs = torch.stack(outputs.scores, dim=0).log_softmax(dim=-1)  # (steps, vocab)

        return LLMResponse(
            text=text,
            tokens_used=tokens_used,
            logprobs=logprobs,
            hidden_states=hidden,
        )

    def _generate_vllm(
        self, prompt, max_new_tokens, temperature,
        return_hidden, return_logprobs,
    ) -> LLMResponse:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            logprobs=1 if return_logprobs else None,
        )
        results = self._model.generate([prompt], params)
        output = results[0].outputs[0]
        text = output.text
        tokens_used = len(output.token_ids)

        return LLMResponse(
            text=text,
            tokens_used=tokens_used,
            logprobs=None,      # vLLM logprobs require separate extraction
            hidden_states=None, # vLLM does not expose hidden states by default
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    @property
    def tokenizer(self):
        return self._tokenizer
