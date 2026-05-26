from __future__ import annotations

import os
from typing import Any, Dict, List


def _env_disable_thinking() -> bool:
    v = str(os.getenv("EQG_DISABLE_THINKING", "1")).strip().lower()
    return v not in {"0", "false", "no", "off"}


def apply_chat_template_compat(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    **kwargs: Any,
) -> Any:
    """Apply chat template with optional Qwen3 thinking-off flag.

    Behavior:
    - If `EQG_DISABLE_THINKING` is truthy (default), we try passing
      `enable_thinking=False` when caller didn't set it explicitly.
    - If tokenizer doesn't support this kwarg (e.g., older Qwen2.5 paths),
      we automatically retry without it.
    """
    trial_kwargs = dict(kwargs)
    inject = _env_disable_thinking() and ("enable_thinking" not in trial_kwargs)
    if inject:
        trial_kwargs["enable_thinking"] = False

    try:
        return tokenizer.apply_chat_template(messages, **trial_kwargs)
    except TypeError:
        if inject:
            fallback_kwargs = dict(kwargs)
            return tokenizer.apply_chat_template(messages, **fallback_kwargs)
        raise
