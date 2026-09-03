from __future__ import annotations

from typing import Protocol


class LLMClientError(RuntimeError):
    """Vendor-neutral failure raised by LLMClient implementations (transport, auth, timeout)."""


class LLMClient(Protocol):
    """
    Minimal model-transport abstraction.

    An implementation turns (model, system_prompt, user_prompt) into raw response
    text. It knows nothing about ATLAS domain concepts; it only moves text to a
    model and back. Concrete vendor clients live in their own modules and are
    injected wherever an LLMClient is required.
    """

    def complete(self, *, model: str, system_prompt: str, user_prompt: str) -> str:
        ...