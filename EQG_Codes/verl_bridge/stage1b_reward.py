"""
Stage 1b Reward Function: V_align only (no V_edu).

Goal: Optimize question-KC alignment without expensive KT simulation.
Method: Compute V_align using forced (c,d) from dataset, not generated values.
Penalty: question_text == KC name → 0.0 (prevent reward hacking).
"""

from __future__ import annotations

import os
import sys
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Optional

EQG_ROOT = Path(__file__).resolve().parents[1]
if os.getcwd() != str(EQG_ROOT):
    os.chdir(EQG_ROOT)
if str(EQG_ROOT) not in sys.path:
    sys.path.append(str(EQG_ROOT))

from verifier.inference import VerifierScorer
from config.config import load_config

_VERIFIER_SCORER: Optional[VerifierScorer] = None
_PARSE_FAILS = 0
_PARSE_LOG_INTERVAL = 5
_VERBOSE_REWARD = bool(int(os.getenv("EQG_REWARD_VERBOSE", "0")))
_SCORE_WINDOW: list[float] = []
_STAT_INTERVAL = int(os.getenv("EQG_REWARD_STAT_INTERVAL", "20"))
_MEDIUM_ONLY = bool(
    int(os.getenv("EQG_STAGE1B_MEDIUM_ONLY", os.getenv("EQG_MEDIUM_ONLY", "0")))
)
cfg = load_config()


def _get_verifier_scorer() -> VerifierScorer:
    """Lazy init verifier scorer."""
    global _VERIFIER_SCORER
    if _VERIFIER_SCORER is None:
        # device=cpu avoids CUDA conflicts when the reward runs in the async worker.
        print(f"[stage1b_reward] Initializing verifier scorer (device=cpu)", flush=True)
        _VERIFIER_SCORER = VerifierScorer(device="cpu")
    return _VERIFIER_SCORER


def _error_payload(
    error: str,
    *,
    student_id: Optional[str] = None,
    kc: str = "",
    difficulty: str = "",
    question_text: str = "",
) -> Dict[str, Any]:
    """Return consistent schema on failure."""
    return {
        "score": 0.0,
        "align_value": 0.0,
        "question_text": question_text,
        "knowledge_concept": kc,
        "difficulty": difficulty,
        "student_id": student_id,
        "error": error,
        "parse_ok": False,
    }


def _parse_question(text: str, forced_kc: str, forced_diff: str) -> Optional[Dict[str, Any]]:
    """
    Parse generated continuation from forced prefix.
    
    Forced prefix format:
    {"knowledge_concept": "KC", "difficulty_level": "DIFF",
    
    Model should generate:
    "question_text": "QUESTION"}
    
    We reconstruct the full JSON and parse it. If parse fails, model gets 0.0 score
    to learn proper JSON formatting. Only auto-fix unavoidable issues like Chinese
    punctuation that don't reflect actual formatting problems.
    """
    try:
        # Auto-fix Chinese punctuation before parsing (tokenizer-level issue, not a model format error).
        text_fixed = text.replace('，', ',').replace('"', '"').replace('"', '"')

        kc_json = json.dumps(str(forced_kc), ensure_ascii=False)
        diff_json = json.dumps(str(forced_diff), ensure_ascii=False)
        # Reconstruct the full JSON by prepending the forced fields.
        if _MEDIUM_ONLY:
            full_json = f'{{"knowledge_concept": {kc_json}, {text_fixed}'
        else:
            full_json = (
                f'{{"knowledge_concept": {kc_json}, '
                f'"difficulty_level": {diff_json}, {text_fixed}'
            )

        obj = json.loads(full_json)
        if _MEDIUM_ONLY and "difficulty_level" in obj:
            return None

        question_text = obj.get("question_text")
        if question_text is None:
            return None

        question_text = str(question_text).strip()
        if not question_text:
            return None

        return {"question_text": question_text}

    except json.JSONDecodeError:
        return None
    except Exception:
        return None


def _record_score(score: float, verbose: bool) -> None:
    """Track scores and print stats."""
    _SCORE_WINDOW.append(score)
    if not verbose:
        return
    count = len(_SCORE_WINDOW)
    if _STAT_INTERVAL > 0 and count % _STAT_INTERVAL == 0:
        mean = statistics.mean(_SCORE_WINDOW) if _SCORE_WINDOW else 0.0
        std = statistics.pstdev(_SCORE_WINDOW) if len(_SCORE_WINDOW) > 1 else 0.0
        print(
            f"[stage1b_reward_stats] count={count} mean={mean:.4f} std={std:.4f}",
            flush=True
        )


def _finish(payload: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
    """Log and return payload."""
    _record_score(payload.get("score", 0.0), verbose)
    if verbose:
        err = payload.get("error", "")
        print(
            f"[stage1b_reward] score={payload.get('score', 0.0):.4f} "
            f"align={payload.get('align_value','NA')} "
            f"kc={payload.get('knowledge_concept','')} diff={payload.get('difficulty','')} "
            f"err={err}",
            flush=True,
        )
    return payload


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]],
    **_: Any,
) -> Dict[str, Any]:
    """
    Stage 1b reward: V_align only using forced (c,d).
    
    Args:
        solution_str: Model's generated continuation (question_text completion)
        ground_truth: Contains forced_kc, forced_diff from dataset
        extra_info: Additional metadata
    
    Returns:
        Dict with score (V_align) and metadata
    """
    try:
        verbose = (
            bool(extra_info.get("verbose_reward", _VERBOSE_REWARD))
            if extra_info
            else _VERBOSE_REWARD
        )
        
        meta = ground_truth or {}
        student_id = meta.get("student_id")
        forced_kc = meta.get("forced_kc")
        forced_diff = meta.get("forced_diff")
        
        if not forced_kc or not forced_diff:
            return _finish(
                _error_payload(
                    "missing_forced_cd",
                    student_id=student_id,
                    question_text=solution_str[:256],
                ),
                verbose,
            )
        
        parsed = _parse_question(solution_str, forced_kc, forced_diff)
        if not parsed:
            global _PARSE_FAILS
            _PARSE_FAILS += 1
            if _PARSE_FAILS % _PARSE_LOG_INTERVAL == 1:
                head = solution_str.strip().replace("\n", " ")[:200]
                print(
                    f"[stage1b_reward_parse] failures={_PARSE_FAILS} | head={head}",
                    flush=True
                )
            return _finish(
                _error_payload(
                    "unparsable_question",
                    student_id=student_id,
                    kc=forced_kc,
                    difficulty=forced_diff,
                    question_text=solution_str[:256],
                ),
                verbose,
            )
        
        question_text = parsed["question_text"]

        # Anti-reward-hacking: penalize if the model just emits the KC name.
        if question_text.strip().lower() == forced_kc.strip().lower():
            return _finish(
                {
                    "score": 0.0,
                    "align_value": 0.0,
                    "question_text": question_text,
                    "knowledge_concept": forced_kc,
                    "difficulty": forced_diff,
                    "student_id": student_id,
                    "error": "question_equals_kc",
                    "parse_ok": True,
                },
                verbose,
            )
        
        verifier = _get_verifier_scorer()
        v_align = verifier.score_alignment(
            context=question_text,
            concept=forced_kc,
            difficulty=forced_diff,
        )

        # Stage 1b has no V_edu; V_align is the final score.
        final_score = v_align
        
        return _finish(
            {
                "score": final_score,
                "align_value": v_align,
                "question_text": question_text,
                "knowledge_concept": forced_kc,
                "difficulty": forced_diff,
                "student_id": student_id,
                "error": "",
                "parse_ok": True,
            },
            verbose,
        )
        
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"[stage1b_reward_exception] {type(exc).__name__}: {exc}\n{tb}", flush=True)
        return _finish(
            _error_payload(
                f"exception:{type(exc).__name__}",
                student_id=meta.get("student_id") if meta else None,
                question_text=solution_str[:256],
            ),
            verbose,
        )
