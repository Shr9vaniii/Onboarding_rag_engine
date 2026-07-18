"""Shared inference helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

CHAT_ASSISTANT_MARKERS = (
    "<|start_header_id|>assistant<|end_header_id|>",
    "<|im_start|>assistant",
)


def load_dotenv(env_path: Path | None = None) -> None:
    path = env_path or ENV_PATH
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def extract_assistant_reply(decoded: str) -> str:
    """Extract assistant text from a decoded chat completion string."""
    text = decoded.strip()
    for marker in CHAT_ASSISTANT_MARKERS:
        if marker in text:
            text = text.split(marker)[-1].strip()
            break
    else:
        # Prefer splitting on role headers over bare "assistant" word matches.
        header_match = re.search(
            r"(?:^|\n)\s*assistant\s*\n",
            text,
            flags=re.IGNORECASE,
        )
        if header_match:
            text = text[header_match.end() :].strip()
        elif "**Direct Answer:**" in text:
            text = text[text.index("**Direct Answer:**") :]

    if text.startswith(":"):
        text = text[1:].strip()
    return text
