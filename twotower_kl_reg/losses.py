"""KL-regularized MultipleNegativesRankingLoss.

Adds a penalty that discourages the LoRA-adapted model's in-batch similarity
distribution from drifting away from the frozen base model's distribution on
the same batch — the same mechanism RLHF-style fine-tunes use to keep a model
close to a reference policy, adapted here to a bi-encoder's softmax-over-
candidates instead of a token distribution. Motivation: every custom-loop
two-tower run in this project so far (`run_001`, `twotower_split/split_001`,
`twotower_field_gate/field_gate_001`) overfits fast — dev recall@1 peaks at
epoch 1-2 and degrades every epoch after despite training loss still falling.
A leash to the frozen model's own judgment directly targets that failure
mode, without needing any new data.

Design choice, stated for review: the KL term is computed over the *exact*
same in-batch joint softmax the main loss uses (``query_to_doc`` direction,
every positive + negative in the local batch as one softmax, same ``scale``)
— not a simplified per-row version — so the penalty regularizes the actual
quantity being optimized rather than an approximation of it. Only the
single-GPU, non-distributed, single-direction, joint-partition case
(``gather_across_devices=False``, ``directions=("query_to_doc",)``,
``partition_mode="joint"``) is implemented, matching every configuration this
project has ever trained with (`MultipleNegativesRankingLoss`'s pure
defaults) — a distributed or multi-direction run would need extending this,
not silently mis-scoring.

The frozen distribution is obtained by toggling the *same* PEFT-wrapped
model's adapter off (``disable_adapters()`` / ``enable_adapters()``, from
``transformers.integrations.peft.PeftAdapterMixin``, present on
``model[0].auto_model`` once ``twotower.train.add_lora_adapter`` has called
``model.add_adapter(...)``) rather than loading a second model instance —
cheap, and guarantees the "frozen" comparison is against the exact weights
this adapter started from, with no risk of a mismatched config or revision.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from sentence_transformers import util
from torch import Tensor, nn

try:
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
except ImportError:  # ST 3.x
    from sentence_transformers.losses import MultipleNegativesRankingLoss


class KLRegularizedMNRL(nn.Module):
    """MultipleNegativesRankingLoss(model, scale) + kl_weight * KL(frozen ‖ adapted).

    The main term is delegated to the library's own
    ``MultipleNegativesRankingLoss`` unchanged — the identical object
    `top1_ctrl` trained with, same ``scale`` — so this class adds exactly one
    new term rather than reimplementing the proven baseline loss.
    """

    def __init__(
        self,
        model,
        *,
        scale: float = 20.0,
        kl_weight: float = 0.5,
        similarity_fct=util.cos_sim,
    ) -> None:
        super().__init__()
        self.model = model
        self.scale = scale
        self.kl_weight = kl_weight
        self.similarity_fct = similarity_fct
        # Unmodified library loss — top1_ctrl's exact main term.
        self.mnrl = MultipleNegativesRankingLoss(model=model, scale=scale)

    def _auto_model(self):
        return self.model[0].auto_model

    def _embed(self, sentence_features: Iterable[dict[str, Tensor]]) -> list[Tensor]:
        return [self.model(f)["sentence_embedding"] for f in sentence_features]

    def _frozen_embed(self, sentence_features: list[dict[str, Tensor]]) -> list[Tensor]:
        auto = self._auto_model()
        auto.disable_adapters()
        try:
            with torch.no_grad():
                return self._embed(sentence_features)
        finally:
            auto.enable_adapters()

    def _batch_kl(self, adapted: list[Tensor], frozen: list[Tensor]) -> Tensor:
        """KL(frozen ‖ adapted) over the identical query_to_doc joint softmax
        MultipleNegativesRankingLoss(scale=self.scale) computes — see module
        docstring for the exact scope this replicates."""
        queries_a, docs_a = adapted[0], adapted[1:]
        queries_f, docs_f = frozen[0], frozen[1:]
        docs_all_a = torch.cat(docs_a, dim=0)
        docs_all_f = torch.cat(docs_f, dim=0)

        logits_a = self.similarity_fct(queries_a, docs_all_a) * self.scale
        logits_f = self.similarity_fct(queries_f, docs_all_f) * self.scale

        log_p_adapted = F.log_softmax(logits_a, dim=1)
        p_frozen = F.softmax(logits_f, dim=1)
        # F.kl_div(input=log Q, target=P) computes sum P * (log P - log Q) = KL(P || Q).
        # Here P = frozen (teacher, no grad), Q = adapted (student) — standard
        # distillation direction: penalize the student for under-covering mass
        # the frozen model placed weight on.
        return F.kl_div(log_p_adapted, p_frozen, reduction="batchmean")

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features = list(sentence_features)
        adapted = self._embed(sentence_features)
        main_loss = self.mnrl.compute_loss_from_embeddings(adapted, labels)
        frozen = self._frozen_embed(sentence_features)
        kl = self._batch_kl(adapted, frozen)
        return main_loss + self.kl_weight * kl

    def get_config_dict(self) -> dict:
        return {
            "scale": self.scale,
            "kl_weight": self.kl_weight,
            "base_loss": "MultipleNegativesRankingLoss",
        }
