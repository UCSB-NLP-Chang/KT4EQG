"""Standard Bayesian Knowledge Tracing (BKT) — per-KC HMM.

Two hidden states: Unmastered (0), Mastered (1).
Parameters per KC:
    p_l0  — P(mastered at t=0)
    p_t   — P(transition: unmastered → mastered)
    p_g   — P(guess: correct | unmastered)
    p_s   — P(slip: incorrect | mastered)

Fitting via Baum-Welch EM on per-KC observation sequences from the dataset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# Numerical guard
_EPS = 1e-6
_CLIP_LO = 0.001
_CLIP_HI = 0.999


@dataclass
class BKTParams:
    p_l0: float = 0.5
    p_t: float = 0.1
    p_g: float = 0.2
    p_s: float = 0.1


DEFAULT_PARAMS = BKTParams()


def _clip(x: float) -> float:
    return max(_CLIP_LO, min(_CLIP_HI, x))


# ------------------------------------------------------------------
# Forward-backward for a single observation sequence
# ------------------------------------------------------------------

def _emission(obs: int, state: int, p_g: float, p_s: float) -> float:
    """P(obs | state)."""
    if state == 1:  # mastered
        return (1.0 - p_s) if obs == 1 else p_s
    else:  # unmastered
        return p_g if obs == 1 else (1.0 - p_g)


def _forward(
    obs_seq: List[int], p_l0: float, p_t: float, p_g: float, p_s: float,
) -> Tuple[List[List[float]], List[float]]:
    """Forward pass. Returns (alpha, scaling).

    alpha[t][s] = P(state=s, o_1:t) (scaled).
    """
    T = len(obs_seq)
    alpha: List[List[float]] = []
    scales: List[float] = []

    # t = 0
    a = [0.0, 0.0]
    a[0] = (1.0 - p_l0) * _emission(obs_seq[0], 0, p_g, p_s)
    a[1] = p_l0 * _emission(obs_seq[0], 1, p_g, p_s)
    s = a[0] + a[1] + _EPS
    alpha.append([a[0] / s, a[1] / s])
    scales.append(s)

    for t in range(1, T):
        a_prev = alpha[t - 1]
        # Transition: 0→0 = 1-p_t, 0→1 = p_t, 1→0 = 0, 1→1 = 1
        a_new = [0.0, 0.0]
        a_new[0] = a_prev[0] * (1.0 - p_t) * _emission(obs_seq[t], 0, p_g, p_s)
        a_new[1] = (a_prev[0] * p_t + a_prev[1]) * _emission(obs_seq[t], 1, p_g, p_s)
        s = a_new[0] + a_new[1] + _EPS
        alpha.append([a_new[0] / s, a_new[1] / s])
        scales.append(s)

    return alpha, scales


def _backward(
    obs_seq: List[int], scales: List[float],
    p_t: float, p_g: float, p_s: float,
) -> List[List[float]]:
    """Backward pass. Returns beta (scaled)."""
    T = len(obs_seq)
    beta: List[List[float]] = [[0.0, 0.0] for _ in range(T)]
    beta[T - 1] = [1.0, 1.0]

    for t in range(T - 2, -1, -1):
        e0 = _emission(obs_seq[t + 1], 0, p_g, p_s)
        e1 = _emission(obs_seq[t + 1], 1, p_g, p_s)
        # beta[t][0] = P(o_{t+2}:T | state_t=0) scaled
        beta[t][0] = ((1.0 - p_t) * e0 * beta[t + 1][0] + p_t * e1 * beta[t + 1][1])
        beta[t][1] = e1 * beta[t + 1][1]  # mastered is absorbing
        s = scales[t + 1] + _EPS
        beta[t][0] /= s
        beta[t][1] /= s

    return beta


def _posteriors(
    alpha: List[List[float]], beta: List[List[float]],
) -> List[List[float]]:
    """gamma[t][s] = P(state=s | all observations)."""
    T = len(alpha)
    gamma: List[List[float]] = []
    for t in range(T):
        g = [alpha[t][0] * beta[t][0], alpha[t][1] * beta[t][1]]
        s = g[0] + g[1] + _EPS
        gamma.append([g[0] / s, g[1] / s])
    return gamma


def _xi(
    alpha: List[List[float]], beta: List[List[float]],
    obs_seq: List[int], scales: List[float],
    p_t: float, p_g: float, p_s: float,
) -> List[List[List[float]]]:
    """xi[t][i][j] = P(state_t=i, state_{t+1}=j | all obs)."""
    T = len(obs_seq)
    xi_out: List[List[List[float]]] = []
    for t in range(T - 1):
        e0 = _emission(obs_seq[t + 1], 0, p_g, p_s)
        e1 = _emission(obs_seq[t + 1], 1, p_g, p_s)
        vals = [[0.0, 0.0], [0.0, 0.0]]
        vals[0][0] = alpha[t][0] * (1.0 - p_t) * e0 * beta[t + 1][0]
        vals[0][1] = alpha[t][0] * p_t * e1 * beta[t + 1][1]
        vals[1][0] = 0.0  # mastered cannot go back to unmastered
        vals[1][1] = alpha[t][1] * 1.0 * e1 * beta[t + 1][1]
        s = sum(vals[i][j] for i in range(2) for j in range(2)) + _EPS
        xi_out.append([[vals[i][j] / s for j in range(2)] for i in range(2)])
    return xi_out


# ------------------------------------------------------------------
# EM fitting for a single KC
# ------------------------------------------------------------------

def fit_bkt_single_kc(
    obs_sequences: List[List[int]],
    max_iter: int = 50,
    tol: float = 1e-4,
    init_params: Optional[BKTParams] = None,
) -> BKTParams:
    """Fit BKT parameters for one KC using Baum-Welch EM.

    Args:
        obs_sequences: list of observation sequences (each = [0/1, ...])
                       from different students for this KC.
        max_iter: maximum EM iterations.
        tol: convergence tolerance on log-likelihood change.

    Returns:
        Fitted BKTParams.
    """
    # Filter out empty or single-element sequences
    seqs = [s for s in obs_sequences if len(s) >= 2]
    if not seqs:
        # Not enough data; return defaults
        return BKTParams() if init_params is None else init_params

    params = init_params or BKTParams()
    p_l0, p_t, p_g, p_s = params.p_l0, params.p_t, params.p_g, params.p_s

    prev_ll = -float("inf")

    for iteration in range(max_iter):
        # Accumulators for M-step
        acc_l0 = 0.0
        acc_xi_00 = 0.0
        acc_xi_01 = 0.0
        acc_gamma0_correct = 0.0
        acc_gamma0_total = 0.0
        acc_gamma1_incorrect = 0.0
        acc_gamma1_total = 0.0
        total_ll = 0.0

        for obs_seq in seqs:
            alpha, scales = _forward(obs_seq, p_l0, p_t, p_g, p_s)
            beta = _backward(obs_seq, scales, p_t, p_g, p_s)
            gamma = _posteriors(alpha, beta)
            xi_vals = _xi(alpha, beta, obs_seq, scales, p_t, p_g, p_s)

            # Log-likelihood contribution
            total_ll += sum(math.log(s + _EPS) for s in scales)

            # Accumulate
            acc_l0 += gamma[0][1]  # P(mastered at t=0)

            for t in range(len(obs_seq) - 1):
                acc_xi_00 += xi_vals[t][0][0]
                acc_xi_01 += xi_vals[t][0][1]

            for t in range(len(obs_seq)):
                acc_gamma0_total += gamma[t][0]
                acc_gamma1_total += gamma[t][1]
                if obs_seq[t] == 1:
                    acc_gamma0_correct += gamma[t][0]
                else:
                    acc_gamma1_incorrect += gamma[t][1]

        # Check convergence
        if abs(total_ll - prev_ll) < tol:
            break
        prev_ll = total_ll

        # M-step
        n_seqs = len(seqs)
        p_l0 = _clip(acc_l0 / n_seqs)
        denom_t = acc_xi_00 + acc_xi_01 + _EPS
        p_t = _clip(acc_xi_01 / denom_t)
        p_g = _clip(acc_gamma0_correct / (acc_gamma0_total + _EPS))
        p_s = _clip(acc_gamma1_incorrect / (acc_gamma1_total + _EPS))

    return BKTParams(p_l0=p_l0, p_t=p_t, p_g=p_g, p_s=p_s)


# ------------------------------------------------------------------
# Fit all KCs
# ------------------------------------------------------------------

def fit_bkt_all_kcs(
    kc_observations: Dict[str, List[List[int]]],
    max_iter: int = 50,
    tol: float = 1e-4,
    min_sequences: int = 5,
) -> Dict[str, BKTParams]:
    """Fit BKT parameters for every KC.

    Args:
        kc_observations: {kc_name: [[0,1,1,...], ...]} per-student sequences.
        min_sequences: KCs with fewer student sequences use default params.

    Returns:
        {kc_name: BKTParams}
    """
    result: Dict[str, BKTParams] = {}
    for kc, obs_seqs in kc_observations.items():
        if len(obs_seqs) < min_sequences:
            result[kc] = BKTParams()
        else:
            result[kc] = fit_bkt_single_kc(obs_seqs, max_iter=max_iter, tol=tol)
    return result


# ------------------------------------------------------------------
# BKT Inference (online update + prediction)
# ------------------------------------------------------------------

class BKTPredictor:
    """Online BKT predictor for a single student."""

    def __init__(self, kc_params: Dict[str, BKTParams]):
        self.kc_params = kc_params
        self.mastery: Dict[str, float] = {}  # current P(mastered) per KC

    def _get_params(self, kc: str) -> BKTParams:
        return self.kc_params.get(kc, DEFAULT_PARAMS)

    def _get_mastery(self, kc: str) -> float:
        if kc in self.mastery:
            return self.mastery[kc]
        return self._get_params(kc).p_l0

    def update(self, kc: str, correct: int) -> None:
        """Update mastery belief for `kc` given observation."""
        params = self._get_params(kc)
        p_l = self._get_mastery(kc)
        p_g, p_s = params.p_g, params.p_s

        if correct:
            denom = p_l * (1.0 - p_s) + (1.0 - p_l) * p_g + _EPS
            p_l_post = p_l * (1.0 - p_s) / denom
        else:
            denom = p_l * p_s + (1.0 - p_l) * (1.0 - p_g) + _EPS
            p_l_post = p_l * p_s / denom

        # Transition
        p_l_new = p_l_post + (1.0 - p_l_post) * params.p_t
        self.mastery[kc] = p_l_new

    def predict(self, kc: str) -> float:
        """P(correct) for a question on this KC."""
        params = self._get_params(kc)
        p_l = self._get_mastery(kc)
        return p_l * (1.0 - params.p_s) + (1.0 - p_l) * params.p_g

    def process_sequence(self, seq: List[Tuple[str, int]]) -> None:
        """Process a sequence of (kc, correct) observations."""
        for kc, correct in seq:
            self.update(kc, correct)

    def reset(self) -> None:
        """Reset mastery to initial state."""
        self.mastery.clear()


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------

def save_bkt_params(kc_params: Dict[str, BKTParams], path: str) -> None:
    import json
    data = {kc: asdict(p) for kc, p in kc_params.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_bkt_params(path: str) -> Dict[str, BKTParams]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {kc: BKTParams(**v) for kc, v in data.items()}
