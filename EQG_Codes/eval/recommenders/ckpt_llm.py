from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping


def _student_state_text(student_graph: Mapping[str, Any], leaf_kcs: List[str]) -> str:
    lines: List[str] = []
    for kc in leaf_kcs:
        node = student_graph.get(kc)
        p = getattr(node, "posterior1", None) if node is not None else None
        if p is None and isinstance(node, dict):
            p = node.get("posterior1", None)
        if p is None:
            p = 0.5
        lines.append(f"{kc}: {float(p):.6f}")
    return "\n".join(lines)


def _safe_parse_json_obj(raw: str) -> Dict[str, Any] | None:
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        return json.loads(raw[start : end + 1])
    except Exception:
        return None


def resolve_ckpt_load_path(ckpt_path: str) -> str:
    """Resolve a checkpoint folder to a loadable HF directory."""
    if not ckpt_path:
        raise ValueError("recommender_ckpt_path is empty but recommender_type=ckpt was requested.")

    root = Path(ckpt_path).expanduser().resolve()
    candidates = [
        root / "actor" / "huggingface",
        root / "huggingface",
        root,
    ]
    for cand in candidates:
        if cand.is_dir() and (cand / "config.json").is_file():
            return str(cand)

    searched = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        f"Cannot find a HuggingFace checkpoint under: {root}. "
        f"Checked: {searched} (expect config.json)."
    )


class CkptLLMKCRecommender:
    """Checkpoint-backed LLM recommender that selects KC only; difficulty is fixed outside."""

    def __init__(self, ckpt_path: str, device: str, max_attempts: int = 3):
        from models.base_llm import BaseLLM

        load_path = resolve_ckpt_load_path(ckpt_path)
        self.load_path = load_path
        self.model = BaseLLM(model_name=load_path, device=device, trainable=False)
        self.max_attempts = max_attempts
        self.system_prompt = (
            "You are a tutor selecting the next practice target concept.\n"
            "Given student mastery over concepts, pick exactly one concept from the candidate list.\n"
            "Return JSON only: "
            '{"knowledge_concept":"...","reason":"..."}'
        )

    def choose_question(
        self,
        *,
        student_graph: Mapping[str, Any],
        leaf_kcs: List[str],
        practice_lookup: Mapping[tuple, Any],
        medium_kc_candidates: List[str],
        fixed_difficulty: str,
        decode_kwargs: Dict[str, Any],
        rng: random.Random,
    ) -> tuple[Any, str]:
        state_text = _student_state_text(student_graph, leaf_kcs)
        user_prompt = (
            "Student current mastery:\n"
            f"{state_text}\n\n"
            "Candidate concepts:\n"
            f"{leaf_kcs}\n\n"
            "Pick one concept from the list."
        )
        for _ in range(self.max_attempts):
            raw = self.model.generate_chat(self.system_prompt, user_prompt, **decode_kwargs)
            obj = _safe_parse_json_obj(raw)
            if not obj:
                continue
            kc = str(obj.get("knowledge_concept", "")).strip()
            key = (kc, fixed_difficulty)
            q = practice_lookup.get(key)
            if q is not None:
                return q, raw

        # Fallback to random if parsing/validation repeatedly fails.
        kc = rng.choice(medium_kc_candidates)
        return practice_lookup[(kc, fixed_difficulty)], "[fallback_to_random]"
