# Synthetic-data distribution-matching research

Logged 2026-07-19. Citation counts are OpenAlex `cited_by_count` values
checked on that date; counts vary across indexes and change over time.

## 1. SynAlign

**Paper:** [Few-shot LLM Synthetic Data with Distribution Matching](https://arxiv.org/abs/2502.08661)  
**Published:** The Web Conference 2025 Companion,
[DOI 10.1145/3701716.3715245](https://doi.org/10.1145/3701716.3715245)  
**Code:** [nighood/SynAlign](https://github.com/nighood/SynAlign)  
**Citations:** **3** ([OpenAlex record](https://openalex.org/W4410636854))

SynAlign combines exploration-aware selection of real demonstrations, LLM
reasoning about latent linguistic attributes, and Maximum Mean Discrepancy
(MMD) weighting/resampling of generated examples. It reports improvements
across text-prediction tasks and an online retriever A/B test.

**Dorby relevance:** the most directly applicable complete pipeline for
matching lexical/style distributions. Generate an oversized candidate pool,
then weight or select it against real train-only examples. The public repo and
online test improve reproducibility, but the paper is short, recent, and has
limited independent validation.

## 2. Multi-source synthetic data

**Paper:** [Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](https://aclanthology.org/2026.findings-acl.360/)  
**Published:** Findings of ACL 2026,
[PDF](https://aclanthology.org/2026.findings-acl.360.pdf),
[DOI 10.18653/v1/2026.findings-acl.360](https://doi.org/10.18653/v1/2026.findings-acl.360)  
**Code:** [maxschaffelder/synthetic_data_diversity](https://github.com/maxschaffelder/synthetic_data_diversity)  
**Citations:** **0** ([OpenAlex record](https://openalex.org/W4416437308))

The paper finds that mixing synthetic data from multiple source models reduces
distribution collapse and better preserves lexical diversity than relying on
one source model. Larger generator models also produced more lexical diversity.

**Dorby relevance:** useful evidence for generating Arm C with multiple model
families, but it is a design principle rather than a distribution-matching
pipeline. It was published this month, so zero citations is not evidence
against quality; there has been almost no time for independent replication.

## 3. Grounding, taxonomy generation, and discriminator filtering

**Paper:** [Generating Faithful Synthetic Data with Large Language Models: A Case Study in Computational Social Science](https://arxiv.org/abs/2305.15041)  
**Published:** arXiv preprint, 2023  
**Code:** [epfl-dlab/faithful-data-gen](https://github.com/epfl-dlab/faithful-data-gen)  
**Citations:** **14** in OpenAlex
([record](https://openalex.org/W4378465287)); Semantic Scholar showed **46**
on 2026-07-19, illustrating substantial index differences.

The study compares grounding generation in real examples, taxonomy-based
generation, and filtering with a real-vs-synthetic classifier. Grounding was
best in its sarcasm-detection case study. Its discriminator explicitly removes
examples carrying synthetic-only lexical artifacts.

**Dorby relevance:** highly relevant to the candidate-only artifact failure.
It supports a held-out real-vs-synthetic discriminator and rejecting easily
detected synthetic profiles. The method is older and more cited, but its
evidence is from one social-science classification setting and the work remains
a preprint.

## 4. ToEdit

**Paper:** [How to Synthesize Text Data without Model Collapse?](https://arxiv.org/abs/2412.14689)  
**Published:** ICML 2025,
[PMLR paper](https://proceedings.mlr.press/v267/zhu25d.html)  
**Citations:** **0** ([OpenAlex record](https://openalex.org/W4405627756))

The paper identifies coverage narrowing and n-gram over-concentration in fully
generated text. ToEdit instead makes constrained token-level edits to
human-produced text, creating semi-synthetic data while retaining much of the
source distribution. It includes theoretical analysis and experiments across
pretraining, continual pretraining, and supervised fine-tuning.

**Dorby relevance:** strongest evidence that preserving real text and editing
it minimally is safer than generating whole profiles. It is less directly
usable because Dorby needs new coherent structured people and labeled
relationships, not isolated token substitutions.

## Maturity assessment

1. **Most academically vetted: ToEdit.** Main-track ICML publication,
   theoretical justification, and broad experiments. Its fit to Dorby is only
   partial.
2. **Most established practical evidence: Faithful Data Generation.** Oldest,
   most cited, public implementation, and directly tests grounding plus
   discriminator filtering. Its preprint-only status and narrow case study are
   limitations.
3. **Best direct fit for Dorby: SynAlign.** Peer-reviewed, public code, and
   distribution matching plus an online retriever test. Independent evidence
   is still limited.
4. **Useful corroboration, least mature: Synthetic Eggs.** Peer-reviewed ACL
   Findings and public code, but extremely new and not a complete generation
   method.

## Recommended Dorby approach

Use a hybrid rather than treating one paper as settled:

1. Ground every generation in stylistically matched real train examples.
2. Generate an oversized pool with multiple model families.
3. Preserve and monitor field-level length, n-gram, TF-IDF, lexical-diversity,
   and embedding distributions.
4. Apply SynAlign-style MMD weighting or subset selection.
5. Reject samples detected as synthetic by a held-out discriminator.
6. Keep the candidate-only label probe near chance and retain Arm A
   (real-only) as the decision baseline.

