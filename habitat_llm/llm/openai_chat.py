#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import logging
import os
from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from habitat_llm.llm.base_llm import BaseLLM, Prompt


logger = logging.getLogger(__name__)


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
