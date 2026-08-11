"""The three MoE diagnostics. Run these before believing any accuracy number.

The professor's slide is blunt about why: *do not trust MoE only because the
architecture has experts — check whether specialization happened.* An MoE whose
gate collapsed is one model wearing a costume, and it will still report a score.

Diagnostic 1 and 2 are from the slides. Diagnostic 3 is added for this dataset
specifically and is the one most likely to catch a fake win here:

    3. Routing vs seeker identity. On the ``rrf_002`` batch, seeker identity
       alone — no text at all — predicted the label at 0.687 AUC, higher than
       every real model in this project, because 12 of 40 seekers rejected every
       candidate. A gate is structurally able to learn that shortcut. If the
       chosen expert is predictable from *which seeker* a pair belongs to, the
       gate learned "this person says no to everything" and not matching.

The professor's setting (click-rate, many users) does not need Diagnostic 3.
This one does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class GateDiagnostics:
    """Per-task diagnostics computed from gate weights of shape (N, n_tasks, n_experts)."""

    expert_usage: list[list[float]]
    mean_gate_entropy: list[float]
    max_entropy: float
    collapse_ratio: list[float]
    seeker_routing_mi: list[float]
    seeker_routing_mi_normalized: list[float]
    #: Mean normalized MI under random routing with the same marginal. The raw MI
    #: above is uninterpretable without this.
    seeker_routing_null_mean: list[float]
    seeker_routing_null_p95: list[float]
    #: observed - null_mean. This is the number to actually look at.
    seeker_routing_excess: list[float]
    #: Fraction of null draws at least as extreme as observed.
    seeker_routing_p_value: list[float]
    n_examples: int
    n_experts: int
    n_seekers: int
    warnings: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "expert_usage": self.expert_usage,
            "mean_gate_entropy": self.mean_gate_entropy,
            "max_entropy": self.max_entropy,
            "collapse_ratio": self.collapse_ratio,
            "seeker_routing_mi": self.seeker_routing_mi,
            "seeker_routing_mi_normalized": self.seeker_routing_mi_normalized,
            "seeker_routing_null_mean": self.seeker_routing_null_mean,
            "seeker_routing_null_p95": self.seeker_routing_null_p95,
            "seeker_routing_excess": self.seeker_routing_excess,
            "seeker_routing_p_value": self.seeker_routing_p_value,
            "n_examples": self.n_examples,
            "n_experts": self.n_experts,
            "n_seekers": self.n_seekers,
            "warnings": self.warnings,
        }


def _mutual_information(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """MI(a; b) in nats, plus MI normalized by H(a).

    ``a`` is the routing decision (argmax expert), ``b`` the seeker id. Both are
    integer-coded. Normalizing by H(a) gives the *fraction* of the routing
    decision explained by seeker identity.

    **This raw number is not interpretable on its own here, and that is not a
    caveat — it is the main thing to know about it.** With 111 rows spread over
    75 seekers, most seekers appear once, so knowing the seeker nearly determines
    the row and therefore nearly determines its routing. Measured on this data,
    *uniformly random* routing scores a normalized MI of ~0.71 (p95 ~0.76). An
    observed 0.82 is barely above that floor, not "82% seeker-driven."

    So always read this against the permutation null that ``compute`` builds —
    ``excess`` and ``p_value``, not ``seeker_routing_mi_normalized``.
    """
    n = len(a)
    if n == 0:
        return 0.0, 0.0
    a_vals, a_idx = np.unique(a, return_inverse=True)
    b_vals, b_idx = np.unique(b, return_inverse=True)
    if len(a_vals) < 2:
        # No routing variation at all: nothing to explain, MI is 0 by definition.
        return 0.0, 0.0

    joint = np.zeros((len(a_vals), len(b_vals)))
    np.add.at(joint, (a_idx, b_idx), 1.0)
    joint /= n
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)

    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())
    h_a = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    return mi, (mi / h_a if h_a > 1e-12 else 0.0)


def _routing_null(
    routing: np.ndarray,
    seeker_codes: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[float, float, float]:
    """(mean, p95, p_value) of normalized MI under reshuffled routing.

    Shuffling the routing labels preserves their marginal distribution and the
    seeker structure, and destroys only the association between them — so this
    is the distribution of "how much MI do I get for free at this sample size".
    """
    rng = np.random.default_rng(seed)
    observed = _mutual_information(routing, seeker_codes)[1]
    draws = np.empty(n_permutations)
    shuffled = routing.copy()
    for i in range(n_permutations):
        rng.shuffle(shuffled)
        draws[i] = _mutual_information(shuffled, seeker_codes)[1]
    p = float((draws >= observed).mean())
    return float(draws.mean()), float(np.percentile(draws, 95)), p


def compute(
    gates: np.ndarray,
    seeker_ids: Sequence[str],
    *,
    collapse_threshold: float = 0.80,
    entropy_floor_frac: float = 0.05,
    #: Threshold on *excess* MI over the permutation null, not on raw MI.
    mi_excess_threshold: float = 0.15,
    mi_p_value_threshold: float = 0.05,
    n_permutations: int = 400,
    seed: int = 0,
) -> GateDiagnostics:
    """Compute all three diagnostics.

    ``gates``: (N, n_tasks, n_experts) softmax weights.
    ``seeker_ids``: length-N seeker identity per row.
    """
    if gates.ndim != 3:
        raise ValueError(f"expected (N, n_tasks, n_experts), got {gates.shape}")
    n, n_tasks, n_experts = gates.shape
    if len(seeker_ids) != n:
        raise ValueError(f"seeker_ids has {len(seeker_ids)} entries, gates has {n} rows")

    max_ent = float(np.log(n_experts))
    codes = {s: i for i, s in enumerate(sorted(set(seeker_ids)))}
    seeker_codes = np.array([codes[s] for s in seeker_ids])

    usage, entropies, collapse, mis, mis_norm = [], [], [], [], []
    null_means, null_p95s, excesses, p_values = [], [], [], []
    warnings: list[str] = []

    for t in range(n_tasks):
        g = gates[:, t, :]

        # Diagnostic 1: average expert usage.
        u = g.mean(axis=0)
        usage.append([float(v) for v in u])
        top = float(u.max())
        collapse.append(top)
        if top >= collapse_threshold:
            warnings.append(
                f"task {t}: COLLAPSE — expert {int(u.argmax())} holds "
                f"{top:.1%} of gate mass (threshold {collapse_threshold:.0%})"
            )

        # Diagnostic 2: per-example gate entropy.
        ent = float(-(g * np.log(g + 1e-12)).sum(axis=1).mean())
        entropies.append(ent)
        if ent >= max_ent * (1.0 - entropy_floor_frac):
            warnings.append(
                f"task {t}: gate is not routing — mean entropy {ent:.3f} is at "
                f"the uniform ceiling {max_ent:.3f}, i.e. it averages experts"
            )

        # Diagnostic 3: is routing just seeker identity? Judged against a
        # permutation null, because the raw MI is saturated at this sample size.
        routing = g.argmax(axis=1)
        mi, mi_norm = _mutual_information(routing, seeker_codes)
        null_mean, null_p95, p_val = _routing_null(
            routing, seeker_codes, n_permutations, seed
        )
        excess = mi_norm - null_mean
        mis.append(mi)
        mis_norm.append(mi_norm)
        null_means.append(null_mean)
        null_p95s.append(null_p95)
        excesses.append(excess)
        p_values.append(p_val)
        if excess >= mi_excess_threshold and p_val <= mi_p_value_threshold:
            warnings.append(
                f"task {t}: routing tracks seeker identity — normalized MI "
                f"{mi_norm:.3f} vs a random-routing null of {null_mean:.3f} "
                f"(excess {excess:+.3f}, p={p_val:.3f}); suspect the per-seeker "
                "base-rate shortcut"
            )

    return GateDiagnostics(
        expert_usage=usage,
        mean_gate_entropy=entropies,
        max_entropy=max_ent,
        collapse_ratio=collapse,
        seeker_routing_mi=mis,
        seeker_routing_mi_normalized=mis_norm,
        seeker_routing_null_mean=null_means,
        seeker_routing_null_p95=null_p95s,
        seeker_routing_excess=excesses,
        seeker_routing_p_value=p_values,
        n_examples=n,
        n_experts=n_experts,
        n_seekers=len(codes),
        warnings=warnings,
    )


def render(diag: GateDiagnostics, task_names: Sequence[str]) -> str:
    """Plain-text report, including the barplot the slides ask for."""
    lines = [
        f"gate diagnostics — {diag.n_examples} examples, "
        f"{diag.n_experts} experts, {diag.n_seekers} distinct seekers",
    ]
    for t, name in enumerate(task_names[: len(diag.expert_usage)]):
        lines.append(f"\n  task '{name}':")
        lines.append("    1. average expert usage (collapse if one dominates)")
        for m, v in enumerate(diag.expert_usage[t]):
            bar = "#" * int(round(v * 40))
            lines.append(f"       expert {m}  {v:6.1%}  {bar}")
        ent = diag.mean_gate_entropy[t]
        lines.append(
            f"    2. mean gate entropy  {ent:.4f} / {diag.max_entropy:.4f} max "
            f"({ent / diag.max_entropy:.0%} of uniform — lower = more decisive)"
        )
        lines.append(
            f"    3. routing vs seeker  normalized MI "
            f"{diag.seeker_routing_mi_normalized[t]:.3f} vs random-routing null "
            f"{diag.seeker_routing_null_mean[t]:.3f} (p95 {diag.seeker_routing_null_p95[t]:.3f})"
        )
        lines.append(
            f"                          EXCESS {diag.seeker_routing_excess[t]:+.3f}  "
            f"p={diag.seeker_routing_p_value[t]:.3f}   <- read this, not the raw MI"
        )
    if diag.warnings:
        lines.append("\n  WARNINGS:")
        lines.extend(f"    - {w}" for w in diag.warnings)
    else:
        lines.append("\n  no diagnostic warnings")
    return "\n".join(lines)
