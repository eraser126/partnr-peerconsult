#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import logging
import os
import random
import re
import time
from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from habitat_llm.llm.base_llm import BaseLLM, Prompt


logger = logging.getLogger(__name__)


_CONTEXT_OVERFLOW_RE = re.compile(
    r"maximum context length is (?P<context>\d+) tokens and your request has "
    r"(?P<input>\d+) input tokens"
)


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
        # Own retry policy explicitly.  The provider-neutral SDK otherwise
        # retries a small, opaque number of times before raising a generic
        # APIConnectionError, which made temporary gateway 503s skip whole
        # PARTNR episodes.
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            max_retries=0,
        )
        self._validate_conf()
        self.verbose = self.llm_conf.verbose
        self.verbose = True
        self.message_history: List[Dict] = []
        self.keep_message_history = self.llm_conf.keep_message_history
        self.max_transient_attempts = int(
            self.llm_conf.get("max_transient_attempts", 6)
        )
        self.retry_base_delay_s = float(self.llm_conf.get("retry_base_delay_s", 3.0))
        self.retry_max_delay_s = float(self.llm_conf.get("retry_max_delay_s", 60.0))
        self.initial_request_delay_s = float(
            self.llm_conf.get("initial_request_delay_s", 0.0)
        )
        self._initial_request_pending = self.initial_request_delay_s > 0
        # The provider reports exact token counts for an over-budget request.
        # Keep enough room for a valid skill call before retrying with a
        # smaller output budget; otherwise surface the genuine overflow rather
        # than repeatedly submitting the same invalid call.
        self.context_overflow_min_output_tokens = int(
            self.llm_conf.get("context_overflow_min_output_tokens", 64)
        )
        self.context_overflow_safety_tokens = int(
            self.llm_conf.get("context_overflow_safety_tokens", 16)
        )
        self._context_output_cap: Optional[int] = None
        # Shared method-level input budget.  V4 and the centralized baseline
        # use the same Qwen config, so neither gets an implicit context-window
        # advantage when the local service is shared by several render jobs.
        self.input_token_budget = int(self.llm_conf.get("input_token_budget", 0))
        self.input_tokenizer_path = self.llm_conf.get("input_tokenizer_path", "")
        self.prompt_head_token_budget = int(
            self.llm_conf.get("prompt_head_token_budget", 3500)
        )
        self.prompt_tail_min_tokens = int(
            self.llm_conf.get("prompt_tail_min_tokens", 1024)
        )
        self._input_tokenizer = None

    def _validate_conf(self):
        if self.generation_params.stream:
            raise ValueError("Streaming not supported")

    @staticmethod
    def _is_retryable_provider_error(error: Exception) -> bool:
        """Return whether an error is likely a short-lived provider outage."""
        if isinstance(error, (APIConnectionError, APITimeoutError)):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code in {429, 500, 502, 503, 504}
        return False

    def _fit_output_budget_after_context_error(self, error, request_params):
        """Return a smaller request when an OpenAI-compatible server reports
        exact input/context token counts; otherwise return ``None``.
        """
        if not isinstance(error, APIStatusError) or error.status_code != 400:
            return None
        match = _CONTEXT_OVERFLOW_RE.search(str(error))
        if match is None:
            return None

        context_limit = int(match.group("context"))
        input_tokens = int(match.group("input"))
        available = context_limit - input_tokens - self.context_overflow_safety_tokens
        previous = int(request_params.get("max_tokens", 0) or 0)
        capped = min(previous, available)
        if capped < self.context_overflow_min_output_tokens or capped >= previous:
            return None

        retry_params = dict(request_params)
        retry_params["max_tokens"] = capped
        self._context_output_cap = capped
        logger.warning(
            "LLM request exceeds context (%d input / %d window); reducing "
            "max_tokens from %d to %d and retrying once.",
            input_tokens,
            context_limit,
            previous,
            capped,
        )
        return retry_params

    def _get_input_tokenizer(self):
        """Lazily load the exact local Qwen tokenizer used for prompt budgets."""
        if self._input_tokenizer is None:
            if not self.input_tokenizer_path:
                raise ValueError(
                    "input_tokenizer_path is required when input_token_budget is set"
                )
            from transformers import AutoTokenizer

            self._input_tokenizer = AutoTokenizer.from_pretrained(
                self.input_tokenizer_path, trust_remote_code=False
            )
        return self._input_tokenizer

    def _chat_token_count(self, messages) -> int:
        tokenizer = self._get_input_tokenizer()
        return len(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )

    def _apply_input_token_budget(self, messages):
        """Keep task instructions and the latest world-state feedback within a
        shared token budget, dropping only stale middle execution history.
        """
        if self.input_token_budget <= 0:
            return messages
        if not messages or not isinstance(messages[-1].get("content"), str):
            logger.warning("LLM input budget skipped for non-text prompt.")
            return messages

        raw_tokens = self._chat_token_count(messages)
        if raw_tokens <= self.input_token_budget:
            logger.info(
                "LLM input budget: raw_tokens=%d sent_tokens=%d budget=%d compacted=False",
                raw_tokens,
                raw_tokens,
                self.input_token_budget,
            )
            return messages

        tokenizer = self._get_input_tokenizer()
        prompt_tokens = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
        if len(prompt_tokens) < 2:
            raise ValueError("Cannot compact an over-budget empty prompt")

        # The prefix contains the task, tool contract, and initial grounding;
        # the suffix contains the newest action result and object state.  The
        # discarded middle consists of stale, repeated ReAct turns.
        head_tokens = min(self.prompt_head_token_budget, len(prompt_tokens) // 2)
        tail_tokens = max(
            self.prompt_tail_min_tokens,
            min(len(prompt_tokens) - head_tokens, self.input_token_budget - head_tokens),
        )
        marker = (
            "\n\n[Earlier execution history was compacted. The task above and "
            "the newest observations/state below are authoritative.]\n\n"
        )
        compacted_messages = list(messages)
        for _ in range(64):
            compacted_prompt = (
                tokenizer.decode(prompt_tokens[:head_tokens])
                + marker
                + tokenizer.decode(prompt_tokens[-tail_tokens:])
            )
            compacted_messages[-1] = dict(messages[-1], content=compacted_prompt)
            sent_tokens = self._chat_token_count(compacted_messages)
            if sent_tokens <= self.input_token_budget:
                logger.info(
                    "LLM input budget: raw_tokens=%d sent_tokens=%d budget=%d "
                    "compacted=True head_tokens=%d tail_tokens=%d",
                    raw_tokens,
                    sent_tokens,
                    self.input_token_budget,
                    head_tokens,
                    tail_tokens,
                )
                return compacted_messages

            reduction = max(128, sent_tokens - self.input_token_budget + 32)
            if tail_tokens > self.prompt_tail_min_tokens:
                tail_tokens = max(self.prompt_tail_min_tokens, tail_tokens - reduction)
            elif head_tokens > self.prompt_tail_min_tokens:
                head_tokens = max(self.prompt_tail_min_tokens, head_tokens - reduction)
            else:
                break

        raise ValueError(
            f"Unable to compact prompt below shared input token budget "
            f"{self.input_token_budget}"
        )

    def _create_completion_with_retry(self, request_params, request_timeout, label: str):
        """Call the provider with bounded, jittered retries for transient faults."""
        if self._initial_request_pending:
            # Agent 1 can opt into this tiny delay through Hydra.  It avoids
            # both PeerConsult agents flooding a small gateway at episode start.
            delay = self.initial_request_delay_s + random.uniform(0.0, 2.0)
            logger.info("LLM %s: staggering first request by %.1fs", label, delay)
            time.sleep(delay)
            self._initial_request_pending = False

        attempts = max(1, self.max_transient_attempts)
        for attempt in range(1, attempts + 1):
            try:
                return self.client.chat.completions.create(
                    **request_params, timeout=request_timeout
                )
            except Exception as error:
                context_retry = self._fit_output_budget_after_context_error(
                    error, request_params
                )
                if context_retry is not None:
                    return self.client.chat.completions.create(
                        **context_retry, timeout=request_timeout
                    )
                if not self._is_retryable_provider_error(error) or attempt == attempts:
                    raise
                delay = min(
                    self.retry_base_delay_s * (2 ** (attempt - 1)),
                    self.retry_max_delay_s,
                )
                # A small jitter prevents synchronized retry storms from array jobs.
                delay += random.uniform(0.0, min(3.0, delay * 0.15))
                logger.warning(
                    "LLM %s: transient %s on attempt %d/%d; retrying in %.1fs",
                    label,
                    type(error).__name__,
                    attempt,
                    attempts,
                    delay,
                )
                time.sleep(delay)

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

        messages = self._apply_input_token_budget(messages)

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
        self._context_output_cap = None
        completion = self._create_completion_with_retry(
            request_params, request_timeout, "initial response"
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
            if self._context_output_cap is not None:
                retry_limit = min(retry_limit, self._context_output_cap)
            if retry_limit > previous_limit:
                retry_params = dict(request_params)
                retry_params["max_tokens"] = retry_limit
                logger.info(
                    "LLM reply was length-truncated; retrying once with max_tokens=%d",
                    retry_limit,
                )
                completion = self._create_completion_with_retry(
                    retry_params, request_timeout, "length-retry response"
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
