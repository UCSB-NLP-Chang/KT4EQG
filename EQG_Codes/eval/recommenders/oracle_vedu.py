from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Mapping, Tuple

from value.value_fn import value_fn


class OracleVEduRecommender:
    """Oracle recommender: scan all candidate KCs and pick max V_edu."""

    def choose_question(
        self,
        *,
        rt: Any,
        student_id: str,
        practice_lookup: Mapping[Tuple[str, str], Any],
        kc_candidates: List[str],
        fixed_difficulty: str,
    ) -> tuple[Any, Dict[str, Any]]:
        t0 = perf_counter()
        best_q = None
        best_score = float("-inf")
        scores: List[Tuple[str, float]] = []

        for kc in kc_candidates:
            q = practice_lookup[(kc, fixed_difficulty)]
            question = {
                "kc": q.kc,
                "difficulty": q.difficulty,
                "question_text": q.question_text,
            }
            score = float(value_fn(question, rt, student_id))
            scores.append((kc, score))
            if score > best_score:
                best_score = score
                best_q = q

        if best_q is None:
            raise RuntimeError("Oracle V_edu failed to pick a question.")

        elapsed = perf_counter() - t0
        top5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
        meta = {
            "oracle_score": best_score,
            "oracle_elapsed_sec": elapsed,
            "oracle_num_kc_scanned": len(kc_candidates),
            "oracle_top5": [{"kc": kc, "v_edu": s} for kc, s in top5],
        }
        return best_q, meta
