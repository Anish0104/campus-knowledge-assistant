import logging
import os
import time
from abc import ABC, abstractmethod

import httpx

from config import (
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MAX_COMPLETION_TOKENS,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
    GROQ_WAIT_BUDGET_SECONDS,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
)


logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract interface for LLM evidence selection providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name, e.g. 'ollama' or 'groq'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Configured model identifier."""
        ...

    @abstractmethod
    def generate_evidence_selection(
        self,
        *,
        system_prompt: str,
        user_content: str,
        schema: dict,
        multi: bool = False,
    ) -> str:
        """
        Request structured evidence selection from the model.

        Returns the raw JSON content string from the model response.
        Raises an appropriate exception if the provider fails, times out,
        cuts off output, refuses, or returns empty/malformed text.
        """
        ...


class OllamaProvider(LLMProvider):
    """Local inference via Ollama's chat API."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.url = (url if url is not None else os.getenv("OLLAMA_URL", OLLAMA_URL)).rstrip("/")
        self.model = model if model is not None else os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
        self.timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("OLLAMA_TIMEOUT_SECONDS", str(OLLAMA_TIMEOUT_SECONDS)))
        )

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    def generate_evidence_selection(
        self,
        *,
        system_prompt: str,
        user_content: str,
        schema: dict,
        multi: bool = False,
    ) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 800 if multi else 400,
            },
        }

        model_family = (
            self.model.rsplit("/", 1)[-1].split(":", 1)[0].casefold()
        )
        if model_family == "qwen3":
            payload["think"] = False

        response = httpx.post(
            f"{self.url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()

        logger.debug(
            "Ollama finished: model=%s mode=%s done_reason=%s tokens=%s",
            self.model,
            "multi" if multi else "single",
            result.get("done_reason", "unknown"),
            result.get("eval_count", "unknown"),
        )

        if result.get("done_reason") == "length":
            raise ValueError(
                "Model output was cut off; answer not displayed."
            )

        message = result.get("message")
        if not isinstance(message, dict):
            raise ValueError("Model returned no message.")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Model returned no answer text.")

        return content


class GroqProvider(LLMProvider):
    """Cloud inference via Groq's OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        api_url: str | None = None,
        max_completion_tokens: int | None = None,
        max_retries: int = 3,
        wait_budget: float | None = None,
    ):
        self.api_key = (
            api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")
        ).strip()
        self.model = (
            model if model is not None else os.getenv("GROQ_MODEL", GROQ_MODEL)
        ).strip()
        self.timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("GROQ_TIMEOUT_SECONDS", str(GROQ_TIMEOUT_SECONDS)))
        )
        self.api_url = (
            api_url if api_url is not None else os.getenv("GROQ_API_URL", GROQ_API_URL)
        ).strip()
        self.max_completion_tokens = (
            max_completion_tokens
            if max_completion_tokens is not None
            else int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", str(GROQ_MAX_COMPLETION_TOKENS)))
        )
        self.max_retries = max_retries
        self.wait_budget = (
            wait_budget
            if wait_budget is not None
            else float(os.getenv("GROQ_WAIT_BUDGET_SECONDS", str(GROQ_WAIT_BUDGET_SECONDS)))
        )

    @property
    def name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self.model

    def generate_evidence_selection(
        self,
        *,
        system_prompt: str,
        user_content: str,
        schema: dict,
        multi: bool = False,
    ) -> str:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        if not self.model:
            raise ValueError("GROQ_MODEL cannot be blank.")

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "evidence_selection",
                    "strict": True,
                    "schema": schema,
                },
            },
            "temperature": 0,
            "max_completion_tokens": self.max_completion_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = None
        total_waited = 0.0

        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 429 and attempt < self.max_retries:
                    remaining_budget = self.wait_budget - total_waited
                    retry_after_header = response.headers.get("Retry-After")
                    delay = None

                    if retry_after_header:
                        try:
                            delay = float(retry_after_header)
                        except (ValueError, TypeError):
                            delay = None

                    # If header specifies delay, never retry earlier than requested
                    if delay is not None:
                        if delay > remaining_budget:
                            raise ValueError(
                                f"Groq rate limit exceeded: Retry-After delay ({delay:.1f}s) "
                                f"exceeds remaining wait budget ({remaining_budget:.1f}s)."
                            )
                        sleep_time = delay
                    else:
                        sleep_time = 2.0 * (2 ** attempt)
                        if sleep_time > remaining_budget:
                            raise ValueError(
                                f"Groq rate limit exceeded: retry delay ({sleep_time:.1f}s) "
                                f"exceeds remaining wait budget ({remaining_budget:.1f}s)."
                            )

                    logger.warning(
                        "Groq rate limit hit (429). Waiting full %.2fs before retry (attempt %d/%d)...",
                        sleep_time,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(sleep_time)
                    total_waited += sleep_time
                    continue

                response.raise_for_status()
                break

            except httpx.HTTPStatusError as error:
                # Do not retry authentication errors (401, 403)
                if error.response.status_code in (401, 403):
                    raise
                if error.response.status_code == 429 and attempt < self.max_retries:
                    continue
                raise

        if response is None:
            raise ValueError("No response received from Groq.")

        result = response.json()

        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Groq returned no choices.")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("Groq returned an invalid choice.")

        finish_reason = choice.get("finish_reason")
        logger.debug(
            "Groq finished: model=%s mode=%s finish_reason=%s",
            self.model,
            "multi" if multi else "single",
            finish_reason,
        )

        if finish_reason != "stop":
            raise ValueError(
                f"Groq response did not complete normally ({finish_reason}); "
                "answer not displayed."
            )

        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Groq returned no message.")

        if message.get("refusal"):
            raise ValueError(f"Groq refused request: {message['refusal']}")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Groq returned no answer text.")

        return content


def get_provider(name: str | None = None) -> LLMProvider:
    """Instantiate the active LLM provider."""
    provider_name = (
        name if name is not None else os.getenv("LLM_PROVIDER", LLM_PROVIDER)
    ).strip().lower()

    if provider_name == "ollama":
        return OllamaProvider()
    elif provider_name == "groq":
        return GroqProvider()
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider_name!r}. Must be 'ollama' or 'groq'."
        )
