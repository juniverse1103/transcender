"""Harmony template rendering for GPT-OSS chat format."""

from __future__ import annotations
from typing import Dict, List, Optional


def build_harmony_messages(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def apply_harmony_template(
    tokenizer,
    user_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    reasoning_effort: str = "medium",
    system_prompt: str = "You are a helpful assistant.",
    add_generation_prompt: bool = True,
) -> tuple[str, List[Dict[str, str]]]:
    """Render an OpenAI-style message list through the GPT-OSS Harmony template."""
    if messages is None:
        if user_prompt is None:
            raise ValueError("Provide either user_prompt or messages.")
        messages = build_harmony_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support apply_chat_template().")

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=reasoning_effort,
    )
    return prompt, messages
