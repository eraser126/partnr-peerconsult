#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from habitat_llm.llm.base_llm import BaseLLM, Prompt


logger = logging.getLogger(__name__)


_GATEWAY_MODEL_PROBES = set()
_CONTEXT_METADATA_KEYS = {
    "context_length",
    "max_context_length",
    "max_model_len",
    "max_sequence_length",
    "max_tokens",
    "max_output_tokens",
    "model_max_length",
}


def _model_payload(value: Any) -> Any:
    """Return model metadata as plain Python objects without logging raw headers."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _context_metadata(value: Any, path: str = "") -> Dict[str, Any]:
    """Keep only explicit context-limit fields from gateway model metadata."""
    result: Dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key).lower() in _CONTEXT_METADATA_KEYS and isinstance(
                child, (str, int, float, bool, type(None))
            ):
                result[child_path] = child
            elif isinstance(child, (Mapping, list, tuple)):
                result.update(_context_metadata(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.update(_context_metadata(child, f"{path}[{index}]"))
    return result


def log_completion_usage(completion, attempt: str) -> None:
    """Log provider-reported usage without exposing prompts or credentials."""
    usage = getattr(completion, "usage", None)
    finish_reason = completion.choices[0].finish_reason
    if usage is None:
        logger.info("LLM %s: finish_reason=%s; token usage unavailable", attempt, finish_reason)
        return
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None)
    logger.info(
        "LLM %s: finish_reason=%s prompt_tokens=%s completion_tokens=%s "
        "reasoning_tokens=%s total_tokens=%s",
        attempt,
        finish_reason,
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
        reasoning_tokens,
        getattr(usage, "total_tokens", None),
    )


def generate_message(multimodal_prompt, image_detail="auto"):
    # Converts the multimodal prompt to the OpenAI format.
    content = []
    for prompt_type, prompt_value in multimodal_prompt:
        if prompt_type == "text":
            message_item = {"type": "text", "text": prompt_value}
        else:
            message_item = {
                "type": "image_url",
                "image_url": {
                    "url": prompt_value,
                    "detail": image_detail,
                },
            }
        content.append(message_item)
    return {"role": "user", "content": content}


class OpenAIChat(BaseLLM):
    def __init__(self, conf: DictConfig):
        """
        Initialize the chat model.
        :param conf: the configuration of the language model
        """
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            assert len(api_key) > 0, ValueError("No OPENAI_API_KEY keys provided")
        except Exception:
            raise ValueError("No OPENAI API keys provided")
        try:
            base_url = os.getenv("OPENAI_BASE_URL")
            assert base_url is not None and len(base_url) > 0
        except Exception:
            raise ValueError(
                "No OPENAI_BASE_URL provided. Set it to an OpenAI-compatible "
                "base URL supplied by your API gateway"
            )
        # Use the provider-neutral OpenAI-compatible client.  The API key is
        # deliberately read only from the process environment and is never
        # stored in Hydra configs, Slurm scripts, or experiment outputs.
        self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
        self._validate_conf()
        self.verbose = self.llm_conf.verbose
        self.verbose = True
        self.message_history: List[Dict] = []
        self.keep_message_history = self.llm_conf.keep_message_history
        self.context_diagnostics = bool(self.llm_conf.get("context_diagnostics", False))

    def _gateway_context_probe(self, model: str) -> None:
        """Best-effort metadata probe; OpenAI-compatible APIs need not expose limits."""
        key: Tuple[str, str] = (str(self.client.base_url), model)
        if key in _GATEWAY_MODEL_PROBES:
            return
        _GATEWAY_MODEL_PROBES.add(key)
        try:
            metadata = _model_payload(self.client.models.retrieve(model))
        except Exception as error:
            logger.info(
                "CONTEXT_DIAGNOSTIC gateway model metadata unavailable for model=%s: %s",
                model,
                type(error).__name__,
            )
            return
        limits = _context_metadata(metadata)
        if limits:
            logger.info(
                "CONTEXT_DIAGNOSTIC gateway model=%s advertised_context_limits=%s",
                model,
                limits,
            )
        else:
            logger.info(
                "CONTEXT_DIAGNOSTIC gateway model=%s does not advertise a context-window field via /models/{model}",
                model,
            )

    def _prompt_token_count(self, messages: List[Dict]) -> Tuple[Optional[int], str]:
        """Count the serialized chat request with the official Qwen tokenizer if possible."""
        tokenizer_model = self.llm_conf.get("context_tokenizer_model", None)
        if tokenizer_model:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_model,
                    local_files_only=bool(
                        self.llm_conf.get("context_tokenizer_local_files_only", False)
                    ),
                )
                token_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                if hasattr(token_ids, "tolist"):
                    token_ids = token_ids.tolist()
                if token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[0]
                return len(token_ids), f"official-tokenizer:{tokenizer_model}"
            except Exception as error:
                logger.info(
                    "CONTEXT_DIAGNOSTIC official tokenizer unavailable for model=%s: %s; falling back to estimate",
                    tokenizer_model,
                    type(error).__name__,
                )
        text = "\n".join(
            str(message.get("content", "")) for message in messages
        )
        try:
            import tiktoken

            return len(tiktoken.get_encoding("cl100k_base").encode(text)), "estimate:tiktoken-cl100k"
        except Exception:
            return None, "unavailable"

    def _log_context_diagnostics(
        self, messages: List[Dict], request_params: Dict[str, Any]
    ) -> None:
        if not self.context_diagnostics:
            return
        model = str(request_params["model"])
        self._gateway_context_probe(model)
        prompt_tokens, tokenizer_source = self._prompt_token_count(messages)
        text = "\n".join(str(message.get("content", "")) for message in messages)
        max_tokens = request_params.get("max_tokens", "omitted; gateway default")
        logger.info(
            "CONTEXT_DIAGNOSTIC request model=%s messages=%d utf8_bytes=%d "
            "prompt_tokens=%s tokenizer=%s requested_max_tokens=%s",
            model,
            len(messages),
            len(text.encode("utf-8")),
            prompt_tokens,
            tokenizer_source,
            max_tokens,
        )

    def _validate_conf(self):
        if self.generation_params.stream:
            raise ValueError("Streaming not supported")

    # @retry(Timeout, tries=3)
    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
        request_timeout: int = 40,
    ):
        """
        Generate a response autoregressively.
        :param prompt: A string with the input to the language model.
        :param image: Image input
        :param stop: A string that determines when to stop generation
        :param max_length: The max number of tokens to generate.
        :param request_timeout: maximum time before timeout.
        :param generation_args: contains arguments like the grammar definition. We don't use this here
        """

        params = OmegaConf.to_object(self.generation_params)

        # Override stop if provided
        if stop is None and len(self.generation_params.stop) > 0:
            stop = self.generation_params.stop
        params["stop"] = stop

        # Override max_length if provided
        if max_length is not None:
            params["max_tokens"] = max_length

        messages = self.message_history.copy()
        # Add system message if no messages
        if len(messages) == 0:
            messages.append({"role": "system", "content": self.llm_conf.system_message})

        # `request_timeout` is a local client option rather than a Chat
        # Completions payload field.  Sending it to an OpenAI-compatible
        # gateway makes the request invalid.
        request_timeout = params.pop("request_timeout", request_timeout)
        if type(prompt) is str:
            # Add current message
            messages.append({"role": "user", "content": prompt})

        else:
            # Multimodal prompt
            image_detail = "low"  # high/low/auto
            messages.append(generate_message(prompt, image_detail=image_detail))

        # Remove optional values the API rejects when null.  Keep the
        # remaining standard Chat Completions controls (temperature, top_p,
        # penalties, max_tokens, and stop) provider-visible and reproducible.
        request_params = {
            "model": params["model"],
            "messages": messages,
            **{
                key: value
                for key, value in params.items()
                if key != "model" and value is not None
            },
        }
        self._log_context_diagnostics(messages, request_params)
        completion = self.client.chat.completions.create(
            **request_params, timeout=request_timeout
        )
        self.last_finish_reason = completion.choices[0].finish_reason
        log_completion_usage(completion, "initial response")

        # Some routed reasoning models use their whole completion budget
        # before emitting the required PARTNR skill call.  Retry exactly once
        # only when the provider explicitly reports length truncation; normal
        # replies make one request as before.  This adapter is shared by
        # baseline and PeerConsult, so their interface treatment stays equal.
        if self.last_finish_reason == "length":
            previous_limit = int(request_params.get("max_tokens", 0) or 0)
            retry_limit = min(max(1024, previous_limit * 2), 2048)
            if retry_limit > previous_limit:
                retry_params = dict(request_params)
                retry_params["max_tokens"] = retry_limit
                logger.info(
                    "LLM reply was length-truncated; retrying once with max_tokens=%d",
                    retry_limit,
                )
                completion = self.client.chat.completions.create(
                    **retry_params, timeout=request_timeout
                )
                self.last_finish_reason = completion.choices[0].finish_reason
                log_completion_usage(completion, "length-retry response")

        text_response = completion.choices[0].message.content or ""
        self.response = text_response

        # Update message history
        if self.keep_message_history:
            self.message_history = messages.copy()
            self.message_history.append({"role": "assistant", "content": text_response})

        if stop is not None:
            text_response = text_response.split(stop)[0]
        return text_response
