from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple


class LowestPosteriorRecommender:
    """Baseline recommender: always pick the KC with the lowest posterior1."""

    def choose_question(
        self,
        *,
        student_graph: Mapping[str, Any],
        practice_lookup: Mapping[Tuple[str, str], Any],
        kc_candidates: List[str],
        fixed_difficulty: str,
    ) -> tuple[Any, Dict[str, Any]]:
        best_q = None
        best_kc = None
        best_posterior = float("inf")
        posteriors: List[Tuple[str, float]] = []

        for kc in kc_candidates:
            node = student_graph.get(kc)
            if node is None:
                continue
            p1 = getattr(node, "posterior1", None)
            if p1 is None:
                p1 = 0.0  # treat missing posterior as unmastered
            p1 = float(p1)
            posteriors.append((kc, p1))
            if p1 < best_posterior:
                best_posterior = p1
                best_kc = kc
                best_q = practice_lookup[(kc, fixed_difficulty)]

        if best_q is None:
            raise RuntimeError("LowestPosterior failed to pick a question.")

        bottom5 = sorted(posteriors, key=lambda x: x[1])[:5]
        meta = {
            "chosen_kc": best_kc,
            "chosen_posterior1": best_posterior,
            "num_kc_scanned": len(kc_candidates),
            "bottom5": [{"kc": kc, "posterior1": p} for kc, p in bottom5],
        }
        return best_q, meta
