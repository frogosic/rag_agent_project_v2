import logging
import os
import anthropic
import requests

from functools import lru_cache
from typing import Any, Literal
from anthropic.types import MessageParam, ToolParam
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

ProviderName = Literal["anthropic", "openai", "ollama"]

DEFAULT_LLM_PROVIDER = "anthropic"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"
DEFAULT_OPENAI_MODEL = "gpt-5.1"
DEFAULT_OLLAMA_MODEL = "llama3.1"


class LLMClient:
    """
    Thin wrapper around the active LLM provider.

    Supported providers:
    - anthropic
    - openai
    - ollama

    The rest of the app should depend on this wrapper rather than importing
    provider SDKs directly.
    """

    def __init__(
        self,
        provider: ProviderName | None = None,
        model: str | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        self.provider: ProviderName = provider or self._get_provider_from_env()
        self.model = model or self._get_default_model(self.provider)
        self.ollama_base_url = (
            ollama_base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        )

        self._anthropic_client: anthropic.Anthropic | None = None
        self._openai_client: OpenAI | None = None

        if self.provider == "anthropic":
            self._anthropic_client = anthropic.Anthropic()
        elif self.provider == "openai":
            self._openai_client = OpenAI()
        elif self.provider == "ollama":
            pass
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def complete(
        self,
        messages: list[MessageParam],
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> str:
        """Run a plain text completion."""
        logger.info(
            "Calling LLM provider=%s model=%s",
            self.provider,
            self.model,
        )

        if self.provider == "anthropic":
            return self._complete_anthropic(
                messages=messages,
                max_tokens=max_tokens,
                system=system,
            )

        if self.provider == "openai":
            return self._complete_openai(
                messages=messages,
                max_tokens=max_tokens,
                system=system,
            )

        if self.provider == "ollama":
            return self._complete_ollama(
                messages=messages,
                system=system,
            )

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def complete_with_tool(
        self,
        messages: list[MessageParam],
        tool: ToolParam,
        max_tokens: int = 300,
    ) -> dict[str, Any]:
        """
        Force a tool call and return the tool input.

        Currently implemented only for Anthropic.
        """
        if self.provider != "anthropic":
            raise NotImplementedError(
                "complete_with_tool is currently implemented only for Anthropic."
            )

        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client is not initialized.")

        logger.info(
            "Calling Anthropic model with forced tool: %s",
            self.model,
        )

        response = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=messages,
        )

        for block in response.content:
            if block.type == "tool_use":
                return dict(block.input)

        raise ValueError("LLM response did not contain a tool_use block.")

    def _complete_anthropic(
        self,
        messages: list[MessageParam],
        max_tokens: int,
        system: str | None,
    ) -> str:
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client is not initialized.")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        response = self._anthropic_client.messages.create(**kwargs)

        for block in response.content:
            if block.type == "text":
                return block.text

        raise ValueError("Anthropic response did not contain a text block.")

    def _complete_openai(
        self,
        messages: list[MessageParam],
        max_tokens: int,
        system: str | None,
    ) -> str:
        if self._openai_client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        openai_messages: list[dict[str, str]] = []

        if system:
            openai_messages.append(
                {
                    "role": "system",
                    "content": system,
                }
            )

        for message in messages:
            content = message.get("content")

            if not isinstance(content, str):
                raise ValueError(
                    "OpenAI adapter currently supports string message content only."
                )

            openai_messages.append(
                {
                    "role": message["role"],
                    "content": content,
                }
            )

        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=openai_messages,  # type: ignore[arg-type]
            max_completion_tokens=max_tokens,
        )

        content = response.choices[0].message.content

        if content is None:
            raise ValueError("OpenAI response did not contain text content.")

        return content

    def _complete_ollama(
        self,
        messages: list[MessageParam],
        system: str | None,
    ) -> str:
        ollama_messages: list[dict[str, str]] = []

        if system:
            ollama_messages.append(
                {
                    "role": "system",
                    "content": system,
                }
            )

        for message in messages:
            content = message.get("content")

            if not isinstance(content, str):
                raise ValueError(
                    "Ollama adapter currently supports string message content only."
                )

            ollama_messages.append(
                {
                    "role": message["role"],
                    "content": content,
                }
            )

        response = requests.post(
            f"{self.ollama_base_url}/api/chat",
            json={
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        message = data.get("message", {})
        content = message.get("content")

        if not isinstance(content, str):
            raise ValueError("Ollama response did not contain text content.")

        return content

    @staticmethod
    def _get_provider_from_env() -> ProviderName:
        provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower()

        if provider not in {"anthropic", "openai", "ollama"}:
            raise ValueError(
                "LLM_PROVIDER must be one of: anthropic, openai, ollama. "
                f"Got: {provider}"
            )

        return provider  # type: ignore[return-value]

    @staticmethod
    def _get_default_model(provider: ProviderName) -> str:
        env_model = os.getenv("LLM_MODEL")

        if env_model:
            return env_model

        if provider == "anthropic":
            return DEFAULT_ANTHROPIC_MODEL

        if provider == "openai":
            return DEFAULT_OPENAI_MODEL

        if provider == "ollama":
            return DEFAULT_OLLAMA_MODEL

        raise ValueError(f"Unsupported LLM provider: {provider}")


@lru_cache(maxsize=8)
def get_llm_client(
    provider: ProviderName | None = None,
    model: str | None = None,
) -> LLMClient:
    """Cached factory so we do not create new clients per request."""
    return LLMClient(provider=provider, model=model)
