# Predicting Introduction Acceptance in a Professional Networking System

**Boardy AI × RecSys Course Project**
Ga Wu (Instructor) · Industry Partner: Boardy AI · 2026

---

## Abstract

When a professional networking platform suggests that two people meet, a second question follows: will they actually say yes? This project addresses that question for Boardy AI, a system that recommends professional introductions. All 200 pairs in our dataset were already recommended by the live system — the label is the real human outcome: accepted or declined. Because both classes passed the same topical-relevance check, we are not modeling topic match; we are modeling whether two people will actually connect.

The most important finding is a split: *finding* the right candidate and *predicting* whether the meeting will happen are two sub-tasks that respond to different inputs. Re-weighting the search query against the full profile, on a frozen model with no training, lifts the top-1 retrieval hit rate from 0.18 to 0.30 and top-10 from 0.59 to 0.91. Fine-tuning a large encoder on generated training data pushes accept/decline separation to 0.6446 AUC and 0.6862 on the hardest declined pairs. Adding a term for whether the candidate would also want the seeker further lifts results to 0.7345 on the held-out test set. An LLM asked to decide directly — without even seeing the search query — reaches 0.6451 and is the only approach that does better on hard-to-distinguish declined pairs than easy ones, suggesting it reasons about fit rather than word overlap. Across all results, choosing *what* text to encode consistently outperformed changing the model architecture.

---

## 1. Introduction

Boardy AI matches professionals for introductions. A seeker provides a profile with a `lookingFor` section — a wish list that may cover several different goals at once (investors, co-founders, clients) — and a `searchQuery`, which is a short statement of what they want right now. The system finds candidate matches and surfaces them. Production uses Voyage-4-large, a large semantic embedding model, to score profile similarity.

This project asks a harder question: given that the system already surfaced a candidate as relevant, can we predict whether the human will accept that introduction?

This is different from ordinary search. As the professor's framing puts it: "the `lookingFor` section is a mixture of multiple properties... but if there is a query there, then based on that `lookingFor` and that query, we need to re-schedule the order of the returned results." The wish list reflects long-term interests; the query reflects the moment. Both matter, but for different reasons.

**The label is not topical relevance.** Every pair in the dataset passed production's own relevance filter. The declined pairs are production's false positives — plausible-looking introductions that a human still turned down. That makes the task harder than standard retrieval: there is no easy-negative population, just pairs that all looked good on paper.

**Contributions:**

1. We show that re-weighting the search query relative to the rest of the seeker's profile is the strongest single lever — nearly doubling top-1 hit rate with no training.
2. We show that fine-tuning a shared encoder on generated training data, when that data passes three leakage checks, substantially improves accept/decline separation.
3. We introduce a three-probe leakage gate for generated training data and demonstrate it separated a harmful first batch (99.2% cheatability) from a clean one.
4. We show that scoring fit in both directions — does the candidate also want the seeker? — adds signal beyond a one-directional score.
5. We show that scoring each line of a seeker's wish list separately, rather than the whole text at once, is the most accurate in-training approach on real pairs alone.
6. We document eight architectural changes that did not improve results, and show that in almost every case a simple text-choice change was more effective.

---

## 2. Related Work

**ELABORATION NEEDED** — This section lists the areas of prior work relevant to each contribution. Citations are placeholders.

- **Separate-encoder retrieval (two-tower / bi-encoder).** The dominant shape for large-scale retrieval: one encoder for the query side, one for the candidate side, both producing vectors that can be compared with a dot product. Candidates are encoded offline; only the query needs a live pass at serving time. **ELABORATION NEEDED** — cite DSSM (Huang et al. 2013), DPR (Karpukhin et al. 2020), representative industrial two-tower systems.

- **Reciprocal / two-sided matching recommenders.** Unlike item recommendation, professional introductions require fit in both directions — the seeker wants the candidate, and the candidate's background should appeal to the seeker too. The Fast-Weight Programmer (FWP) paper by Ga Wu proposes dynamic per-user preference memories updated from interaction history. This project implements the static cold-start reduction (no interaction logs exist). **ELABORATION NEEDED** — cite FWP paper and stable reciprocal recommender work.

- **LLMs as raters and judges.** Instead of embedding vectors, give a large language model both profiles and ask it to decide. Used here as an accuracy ceiling and as a labeler for generated training data. **ELABORATION NEEDED** — cite LLM-as-judge literature (Zheng et al. 2023 on MT-Bench etc.).

- **Light-weight model adaptation (LoRA).** Rather than fine-tuning all parameters, low-rank matrices are added to the attention layers and only those are trained. The base model's weights are frozen. At serving time the adapter weights can be merged into the base model at zero extra cost. **ELABORATION NEEDED** — cite LoRA (Hu et al. 2022).

- **Generated training data for retrieval.** Generating synthetic query–document pairs to supplement scarce labeled data. Key risk: the generator can encode the label into the text, teaching the model to read generation artifacts rather than real signal. **ELABORATION NEEDED** — cite GPL (Wang et al. 2022) and related work on synthetic pair generation for dense retrieval.

---

## 3. Problem Setup

### The task

We have 200 real introductions Boardy surfaced to real users. 100 were accepted — the two people actually connected. 100 were declined. The goal is to train on part of this data and correctly predict accept or decline on the rest.

Because every pair passed Boardy's own relevance check, we are not predicting whether two profiles are about the same subject. We are predicting something subtler: given that this introduction looks right on paper, will these two people actually connect? That signal lives in things like seniority fit, timing, the direction of who needs what from whom, and whether the candidate would also want to meet the seeker.

The practical consequence: expect moderate scores across all approaches. Separating accepted from declined *among already-plausible introductions* is much harder than separating relevant from irrelevant candidates in standard search.

### The personalized-search framing

The professor's description of Boardy is the clearest framing: the `lookingFor` section is a multi-part wish list — a user might be looking for investors, co-founders, and clients simultaneously. The `searchQuery` is a short statement of what they want right now. When someone searches "incubators," they are revealing they want seed funding, even though their `lookingFor` says "investors." Both signals matter. The standing wish list tells us who the person is over time; the query tells us who they want today.

This distinction directly motivates the biggest finding in this project: the query and the profile carry different information, and collapsing them into one text loses both.

### The serving constraint

Boardy needs query results in under 100 milliseconds. This rules out:

- Calling an LLM per candidate (takes seconds per call)
- Cross-encoders that must see both sides together (cost scales with the number of candidates scored)
- Remote API calls per query at serving time

The only shape that fits is a **bi-encoder** (two separate encoders): candidates are encoded and stored in advance; only the query needs a live pass at serving time, followed by a fast nearest-neighbor lookup. The serving cost does not grow with the candidate pool. This is the shape all training experiments in this project use.

---

## 4. Data

### The 200 real pairs

Boardy provided 200 introductions with known outcomes: 100 accepted, 100 declined. Each pair contains:

- **Seeker:** profile fields (`positioning`, `background`, `lookingFor`, `notes`, location/preference fields) plus a `searchQuery`
- **Candidate:** the same profile fields, no query

Membership in the accepted or declined file is the label — there is no explicit label field.

The `lookingFor` field often contains multiple distinct asks written as separate paragraphs, one per goal. This becomes relevant in [Section 10](#10-result-6--score-each-ask-separately).

### Split

The 200 pairs are split user-disjointly: no person appears on both sides of the split. The training side has 131 pairs; the held-out test side has 69 pairs. The split is fixed and frozen in `data/synthetic/seed_split.json`.

**The small held-out set is not used for decisions.** See [Section 6](#6-how-we-measure) for why. The numbers reported throughout this paper are from the full 200-pair scoring described next.

### Scoring population

All main results use **all 200 real pairs**: 100 positive queries (accepted introductions) ranked against a pool of 178 unique candidates. This is the population built by `eval_real_full/`. Source: [`docs/eval-real-full-experiment.md`](eval-real-full-experiment.md).

### Production-scale export

Boardy also shared a larger export of ~46,000 seeker–match rows from the live system (referred to here as B-data). This export does not have accept/decline labels — it has status values (pending, accepted via SMS, etc.) that do not map cleanly to the 200-pair label definition. Results on this export are reported in [Section 15.4](#154-on-the-larger-export).

---

## 5. How We Measure

### Separating accepted from declined (pair AUC)

The main classification metric is area under the ROC curve (AUC) for ranking accepted pairs above declined ones. Random guessing scores 0.50; a perfect ranker scores 1.00. We report this on all 200 pairs and separately on two slices of the declined pairs.

### Hard and easy declined pairs

Not all declined pairs are equally difficult. We measure token overlap — the fraction of words shared — between the seeker's text and the candidate's text. Declined pairs with low overlap are **easy**: the model can probably separate them from accepted pairs by noticing the topic mismatch. Declined pairs with high overlap are **hard**: both sides are talking about similar things, so word-counting does not help. The hard slice is the one that matters in production, because Boardy's own relevance filter already removed the obvious mismatches.

Hard-negative AUC scores the model on *all accepted pairs* versus only the *high-overlap declined pairs*. Easy-negative AUC uses the *low-overlap* declined pairs instead. Every strong model in this project does better on easy negatives than hard ones — except the LLM judge.

### Finding the right person (retrieval metrics)

Beyond the binary accept/decline question, we also ask: given the seeker's query, does the model surface the accepted candidate near the top of the ranked list?

- **Top-1 hit rate (R@1):** fraction of queries where the accepted candidate appears in the first position
- **Top-10 hit rate (R@10):** fraction where it appears in the top 10
- **Mean reciprocal rank (MRR):** average of 1/rank across queries — rewards placing the right answer early

These are measured against a pool of 178 unique candidates. A random ranker gets MRR ≈ 0.006 on this pool.

**Caveat on retrieval scores:** the production system selected candidates using the same `searchQuery` we are using for evaluation. This creates partial circularity — a model that simply ranks by query-text overlap would partially replicate production's selections. The accept/decline AUC scores do not have this problem. **ELABORATION NEEDED** — describe the circularity more precisely and the scope of its effect.

### Why the 69-pair held-out set is not the decision population

Early in the project, results were compared on the frozen 69-pair set. After rescoring all models on all 200 pairs, the ranking among strong models largely inverted. Measured by Spearman correlation, the 69-pair held-out set ranks all models reliably (correlation +0.886), but among the top six models the correlation with the all-200 ranking is effectively zero (−0.029). With only 29 positive queries, a single extra correct answer changes top-1 hit rate by 3.4 percentage points, which is larger than most real differences between models. We report 69-pair numbers where they are relevant for reproducibility but make no decisions from them. Source: [`docs/all-200-baseline-sweep.md`](all-200-baseline-sweep.md).

---

## 6. The Starting Point

Before any training, we measured 14 models on the 200-pair scoring population. The bar to beat is Voyage-4-large (0.5726), which is the encoder Boardy's live system uses.

Pair AUC and top-list ranking (MRR) frequently disagree: TF-IDF word-overlap scoring reaches 0.5649 accept/decline AUC but almost never puts the right candidate at position one (MRR 0.131, R@1 0.05). Qwen3-Embedding-8B, a large open-weight model, was initially reported to beat Voyage-4-large on the 69-pair set — that claim was retracted when re-measured on all 200 pairs (pair AUC 0.5529 vs 0.5726, hard-neg AUC 0.4680, below chance). Source: [`docs/all-200-baseline-sweep.md`](all-200-baseline-sweep.md).

| Model | Pair AUC | Hard-neg AUC | Easy-neg AUC | MRR | R@1 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| Voyage-4-nano (frozen) | 0.5593 | 0.5046 | 0.6960 | 0.3171 | 0.18 | 0.59 |
| **Voyage-4-large (production)** | **0.5726** | 0.5422 | 0.6540 | 0.3102 | 0.13 | **0.70** |
| Frozen BERT (bert-base-uncased) | 0.4697 | 0.4108 | 0.6508 | 0.0941 | 0.02 | 0.18 |
| TF-IDF (word-overlap cosine) | 0.5649 | 0.5164 | 0.6848 | 0.1313 | 0.05 | 0.26 |
| BGE-en-ICL (open-weight) | 0.5389 | 0.5226 | 0.5928 | **0.3190** | **0.17** | 0.62 |
| Qwen3-Embedding-8B (open-weight) | 0.5529 | 0.4680 | 0.7208 | 0.2045 | 0.05 | 0.55 |
| E5-Mistral-7B-instruct (open-weight) | 0.4597 | 0.3772 | 0.6144 | 0.1159 | 0.03 | 0.30 |
| NV-Embed-v2 (open-weight, approx.) | 0.4841 | 0.3836 | 0.6608 | 0.0857 | 0.04 | 0.16 |
| zembed-1-embedding (open-weight) | 0.4707 | 0.4864 | 0.4816 | 0.0377 | 0.01 | 0.08 |

*Fine-tuned models and LLM judges appear in their respective sections below. Full table with fine-tunes in Appendix A. Source: [`docs/baseline-results-real200.md`](baseline-results-real200.md).*

**Key observations before any training:**

- Production's own model (Voyage-4-large) leads pair AUC and R@10 but sits sixth on R@1. 
- Frozen BERT is near chance — its 512-token context truncates long profiles, and it has no domain adaptation.
- TF-IDF's accept/decline AUC (0.5649) is higher than most neural models. This reflects an artifact: easy declined pairs (low word overlap with the seeker) are easy to score correctly by counting words. On hard declined pairs (high word overlap) TF-IDF drops to 0.5164, near chance.
- No open-weight model consistently beats production. BGE-en-ICL leads on retrieval (MRR 0.319 vs 0.310) but trails on accept/decline AUC.

---

## 7. Result 1 — Choosing What Text to Encode

### The query is 2% of the seeker's text

When the seeker's full profile and `searchQuery` are concatenated into one string, the query accounts for about 218 of 10,178 characters — roughly 2%. The model has to extract a short, specific signal from a long, general one. Source: [`docs/query-weighted-encoding-experiment.md`](query-weighted-encoding-experiment.md).

### Encoding the query and profile separately, then blending

Instead of concatenating, encode the search query and the full profile into two separate vectors, then combine them: `normalize(α × query_vector + (1−α) × profile_vector)`. No training. Source: [`docs/query-weighted-encoding-experiment.md`](query-weighted-encoding-experiment.md).

Results on all 200 pairs with frozen Voyage-4-nano:

| Approach | Pair AUC | Hard-neg AUC | MRR | R@1 | R@10 |
|---|---:|---:|---:|---:|---:|
| Concatenate (baseline) | 0.5593 | 0.5046 | 0.3171 | 0.18 | 0.59 |
| Profile only (no query) | 0.5424 | 0.4862 | 0.2357 | 0.09 | 0.50 |
| Query only | 0.5530 | **0.5914** | 0.5019 | **0.30** | **0.91** |
| Blend α=0.6 (60% query, 40% profile) | **0.5872** | 0.5818 | 0.4649 | 0.25 | 0.89 |

The query-only encoding doubles retrieval performance (MRR 0.317 → 0.502, R@1 0.18 → 0.30, R@10 0.59 → 0.91) at zero training cost. It also achieves the best hard-negative AUC of any frozen model on all 200 pairs (0.5914).

The blend at α=0.6 gets the best pair AUC overall (0.5872) — blending in some profile context helps classification while keeping most of the retrieval gain.

The reason is structural: when measuring how much a word in the seeker's text discriminates between their match and a random other candidate, query words discriminate at a ratio of 1.29×, while profile words discriminate at only 1.04×. The biography dilutes the ask.

**Serving cost goes down, not up.** With the concatenated seeker string, the model encodes ~2,500 tokens per query. With query-only, it encodes ~55 tokens. The seeker profile vector can be precomputed and cached per-user, so the live path is one short encode plus a nearest-neighbor lookup.

Calibrating α on generated synthetic pairs (which is disjoint from the 200 real pairs) finds α=1.0 (pure query), but the real-pair sweep shows the best overall AUC is at α=0.6. This is an example of **boundary-monotonic calibration failing to find interior optima**: the synthetic-pair curve rises all the way to the boundary, missing the interior improvement on real pairs. Source: [`docs/nomad-drift-experiment.md`](nomad-drift-experiment.md).

The same finding transfers to fine-tuned models — when the same sweep is applied to the best fine-tuned adapter, query-only again produces the best retrieval (MRR 0.508, R@1 0.32) and α=0.6 again produces the best accept/decline AUC (0.613). Source: [`docs/query-weighted-twotower-experiment.md`](query-weighted-twotower-experiment.md).

### Which profile fields carry person-specific information?

A separate experiment embedded each profile field in complete isolation (just that field's text, no other context) and measured how similar the isolated embedding was to the person's whole-profile embedding versus the same field from other people. Source: [`docs/experiment-graphs-index.md`](experiment-graphs-index.md) (field isolation section).

| Field | Similarity to own profile | Similarity to same field, other people |
|---|---:|---:|
| positioning | 0.890 | 0.549 |
| background | 0.850 | 0.580 |
| lookingFor | 0.852 | 0.595 |
| notes | 0.700 | 0.563 |
| introPreferences | 0.731 | 0.680 |
| locationAvailability | 0.450 | **0.676** |
| meetingAndSchedulingPreferences | 0.391 | **0.751** |
| personalPreferences | 0.366 | **0.756** |

The top three fields (`positioning`, `background`, `lookingFor`) stay person-specific even in isolation. The bottom three (`locationAvailability`, `meetingAndSchedulingPreferences`, `personalPreferences`) are more similar to other people's same field than to their own profile — they read as scheduling boilerplate and carry almost no signal about who the person is. Removing them from encoding removes noise.

### Splitting the wish list into separate asks

The `lookingFor` field often contains several distinct paragraphs, one per goal. When the seeker's `lookingFor` is split into those individual paragraphs and the similarity to each is scored separately, taking the maximum score (or a soft weighted version) outperforms treating the whole field as one text. Source: [`docs/lookingfor-sectioning-findings.md`](lookingfor-sectioning-findings.md).

On the 69-pair held-out set with frozen Voyage-4-nano:

| Seeker encoding | Pair AUC | MRR | R@1 |
|---|---:|---:|---:|
| Whole profile (baseline) | 0.5793 | 0.4610 | 0.28 |
| Wish list split, max score | 0.5957 | 0.4934 | 0.35 |
| Wish list split, soft blend (τ=0.05) | 0.5983 | 0.5149 | 0.38 |
| TF-IDF + split wish list hybrid | **0.6483** | 0.4392 | 0.31 |

Splitting on the seeker side helps; splitting on the candidate side hurts (the candidate's profile should be read as a whole). The hybrid with TF-IDF word-overlap scoring reaches the strongest frozen result on the held-out set (0.6483). These numbers are on the 69-pair set and have not been re-scored on all 200 pairs.

---

## 8. Result 2 — Fine-Tuning a Shared Encoder

### Setup

We add a small trainable adapter (LoRA — low-rank matrices attached to the attention layers, with fewer than 1% of total parameters) to a frozen base encoder and train it to rank the accepted candidate above other candidates seen in the same batch. The base model's weights do not change; only the adapter is trained. At serving time the adapter can be merged into the base weights, so serving cost is identical to the frozen model. Source: [`docs/two-tower-fine-tune-plan.md`](two-tower-fine-tune-plan.md).

We tested two base encoders:
- **Voyage-4-nano:** a 347-million-parameter model that can run locally; this is the serving-feasible option
- **Qwen3-Embedding-8B:** an 8-billion-parameter open model; better accuracy, higher serving cost

**ELABORATION NEEDED** — describe LoRA rank (8), training loss (MultipleNegativesRankingLoss), and why that loss was chosen once per-seeker triplets existed.

### Best fine-tuning results (all 200 pairs)

The most accurate fine-tuned model is a Qwen3-Embedding-8B adapter trained on the `voyage-gemini` generated batch (described in [Section 9](#9-result-3--generating-training-data-that-helps)). It reaches the same accept/decline AUC as the LLM judge — the highest of any approach that can produce a ranked list:

| Model | Pair AUC | Hard-neg AUC | MRR | R@1 | R@10 |
|---|---:|---:|---:|---:|---:|
| Qwen fine-tune (voyage-gemini batch) | **0.6446** | **0.6862** | 0.415 | 0.24 | — |
| Nano fine-tune (query→bg+lookingFor) | 0.5983 | 0.6564 | 0.479 | 0.30 | 0.86 |
| Voyage-4-nano (frozen, query-only) | 0.5530 | 0.5914 | 0.502 | **0.30** | **0.91** |
| Voyage-4-large (frozen, production) | 0.5726 | 0.5422 | 0.310 | 0.13 | 0.70 |

*Note: the Qwen fine-tune uses a batch flagged as somewhat leaky by the pre-training checks (candidate-identity prediction AUC 0.758 vs the ~0.49 floor for real data). The nano fine-tune uses the rrf_003 batch, which passed all three probes. See Limitations.*

Source: [`docs/twotower-qwen-voyage-gemini-experiment.md`](twotower-qwen-voyage-gemini-experiment.md), [`docs/twotower-queryonly-back-look-experiment.md`](twotower-queryonly-back-look-experiment.md).

### The best nano fine-tune: train on query vs background+lookingFor

A 105-way sweep over field combinations on the frozen model found that the best seeker text for retrieval is the search query only (no profile), and the best candidate text is background + lookingFor (no positioning or other fields). A nano adapter trained on that pairing — search query on the seeker side, background+lookingFor on the candidate side — becomes the most accurate nano fine-tune. It is the first model where hard-negative AUC (0.6564) exceeds easy-negative AUC (0.570), a pattern previously seen only in the LLM judge. Source: [`docs/twotower-queryonly-back-look-experiment.md`](twotower-queryonly-back-look-experiment.md).

### Batch size is the key training lever

Showing more examples in each training batch consistently improves ranking. With micro-batch size 2 (2 examples per step before gradient accumulation) versus micro-batch size 6, with everything else fixed:

| Micro-batch | Pair AUC | MRR | R@1 |
|---|---:|---:|---:|
| 2 | 0.600 | 0.490 | 0.33 |
| **6** | **0.598** | **0.533** | **0.38** |

*(On the 69-pair held-out set; rrf_003 batch; MultipleNegativesRankingLoss k=1)*

R@1 improves by six points when the model sees more candidates per batch step. Using gradient accumulation instead of larger micro-batches does not replicate this gain — the benefit comes from having more candidates present at once during the forward pass, not just larger effective batch size. Source: [`docs/twotower-rrf-triplet-ablation-experiment.md`](twotower-rrf-triplet-ablation-experiment.md).

### What fine-tuning actually learns

A direct test: train the model on query-only seeker text, then evaluate it with profile-only seeker text at test time (and vice versa). The test-time text choice dominates — whether the query was present *during training* barely matters; what moves the numbers is what text the model is given *at test time*. The adapter teaches the model to encode things better, but the representation choice at inference still outweighs the specialization from training. Source: [`docs/twotower-no-query-experiment.md`](twotower-no-query-experiment.md).

---

## 9. Result 3 — Generating Training Data That Helps

### Why generated data is necessary and risky

The 200 real pairs are far too few to train a large model. A fine-tuned encoder on real data alone (111 train-split pairs, called Arm A) is only marginally better than the frozen model on hard negatives and flat everywhere else. Generated training data could scale this up — but in the first attempt, it made things worse.

### The first batch failed: label leakage

The first 460-pair batch (`batch_500_001`) was generated by giving a language model the label (positive or negative introduction) before asking it to write the candidate's profile. A trivial check confirmed the problem: a simple word-frequency classifier, trained on the *candidate's profile text alone* (no seeker, no query), predicted the label at 99.2% accuracy on this batch. The same classifier on real pairs scores 48.7% — close to random. The generator was writing different kinds of text for accepted versus declined candidates, so the model learned to read those stylistic differences rather than real matching signals. The fine-tune trained on this data scored 0.4845 hard-negative AUC — *below chance*. Source: [`docs/possible-bugs.md`](possible-bugs.md) (bug #4), [`docs/twotower-run-001-results.md`](twotower-run-001-results.md).

The batch was quarantined. The real-only Arm A (111 pairs, zero generated) beat it on every metric using one-fifth the data — proof that the generated data was actively harmful, not just unhelpful.

### Three probes for checking a generated batch

Before using any generated batch for training, we now run three cheap checks:

1. **Candidate-profile-only prediction:** can a word-frequency classifier on the candidate's text alone predict the label? Near-chance (~0.49) is passing; anything substantially above 0.55 signals that the label was written into the text.
2. **Lexical circularity:** can plain word-overlap between the search query and the candidate's text predict the label? If the method that found candidates (retrieval by word overlap) is also what the labels reflect, then training on these labels teaches word overlap, not real fit.
3. **Seeker-identity prediction:** can knowing only which seeker a pair belongs to (not the pair's text) predict the label? If some seekers are accepted on every candidate while others are rejected on every candidate, a classifier learning "seeker identity → label" will win without learning anything about pair fit.

### The RRF + judge pipeline

The clean generation path uses two independent retrieval channels: a large dense embedding model (Qwen3-Embedding-8B) and BM25 keyword search. Their results are fused by weighted reciprocal rank fusion (giving more weight to the dense results). The top retrieved candidates for each seeker are then passed to a Gemini flash-lite LLM judge, which decides accept or decline for each pair. Source: [`docs/rrf-pairing-pipeline.md`](rrf-pairing-pipeline.md).

This pipeline separates retrieval (which model finds candidates) from labeling (which model decides the label), so neither is grading its own output.

Probe results across batches:

| Batch | Candidate-only AUC | Lexical circularity AUC | Seeker-identity AUC |
|---|---:|---:|---:|
| First batch (failed) | **0.992** | — | — |
| rrf_002 (100 profiles) | 0.634 | 0.701 | 0.687 |
| rrf_003 (1,000 profiles) | — | — | ~0.687 |
| voyage-gemini batch (~9,600 profiles) | 0.758 | 0.481 | 0.780 |

*Real-data floor for candidate-only prediction: 0.487–0.532.*

The rrf_003 batch (which produced most training data for the nano fine-tunes) passes candidate-only and lexical circularity. The seeker-identity leakage (0.687) is handled by training within-seeker — triplet training pairs an anchor with a positive and a negative from the same seeker, which cancels per-seeker base rates.

The voyage-gemini batch (used for the Qwen and `voyage_gemini_ctrl` fine-tunes) shows higher candidate-identity leakage (0.758). The Qwen fine-tune's record numbers come from this batch, so that caveat belongs to those numbers. Source: [`docs/twotower-voyage-gemini-ctrl-experiment.md`](twotower-voyage-gemini-ctrl-experiment.md).

### What transfers and what does not

When a model trained on generated pairs is tested on real pairs, the vocabulary (the kinds of words and topics in the profiles) transfers: a word-frequency model fit on generated profiles does about as well on real profiles as one fit on real profiles directly (AUC 0.563 vs 0.566). But the labels do not transfer: a model that learned to predict generated labels from generated text performs *below chance* on real accept/decline outcomes (0.427 AUC). Source: [`docs/moe-rrf003-synthetic-training-findings.md`](moe-rrf003-synthetic-training-findings.md).

The implication: generated data provides real vocabulary diversity, but the label quality is capped by how accurately the generator/judge combination reflects real human accept/decline decisions. The LLM judge on real pairs scores ~0.59 on its hardest pairs — that is the ceiling for anything trained on its labels.

---

## 10. Result 4 — Asking an LLM Directly

### Setup

Instead of comparing embedding vectors, feed both profiles to a large language model and ask whether the introduction is a good match. Score the yes/no decision (and the model's expressed confidence) against the real accept/decline outcome. The search query is deliberately withheld — this tests whether accept/decline is predictable from profile text alone. Source: [`docs/llm-judge-experiment.md`](llm-judge-experiment.md).

### Results

The best prompts. Numbers in the first three rows are from all 200 pairs; calibrated is from the 69-pair held-out set (the only run measured there).

| Prompt variant | Pair AUC | Hard-neg AUC | Easy-neg AUC | Decision accuracy | Population |
|---|---:|---:|---:|---:|---|
| Focused (seeker positioning+lookingFor+query, candidate positioning+background+lookingFor) | **0.6451** | 0.6590 | 0.6570 | 0.5950 | all-200 |
| Naive (complete profiles, no query) | 0.6177 | — | — | 0.6050 | all-200 |
| Forced step-by-step scoring | 0.6100 | 0.6225 | 0.6394 | 0.5700 | all-200 |
| "Calibrated" (told the base rate and that production pre-vetted relevance) | 0.5901 | 0.5879 | 0.5310 | 0.5652 | 69-pair |

The focused prompt — which gives the model the search query plus trimmed fields on each side — reaches the best accept/decline AUC of anything tested in this project (0.6451) and matches the best fine-tune (0.6446) while being limited to the information in a subset of fields.

### The hard-negative inversion

Every embedding model does *worse* on hard declined pairs (high word overlap with the seeker) than on easy declined pairs (low word overlap). The LLM judge is the only approach that flips this — it does *better* on hard declined pairs (0.6590) than on easy ones (0.6570). Smaller open-weight judges tested on this setup show the same pattern.

This is direct evidence that the LLM is reasoning about structural fit rather than word overlap: it correctly handles the pairs that share vocabulary, which is exactly the population that exists in production (since production's own relevance filter already removed the vocabulary mismatches).

### Adding information did not help

The "calibrated" variant told the model that the base rate is 50/50 and that production had already vetted topical relevance. Accept/decline AUC dropped from 0.6177 to 0.5901. The model became more skeptical (yes-rate 56.5% → 30.4%) but not more accurate.

Forcing multi-aspect step-by-step scoring (six weighted dimensions, aggregated in code) also did not help — averaging six independent scores regresses toward the middle, producing more "yes" answers (75.4% vs 55.1%) without being more often right. Source: [`docs/llm-judge-experiment.md`](llm-judge-experiment.md).

### Not deployable, but useful

An LLM call per candidate takes seconds. The 100 ms budget makes this architecture infeasible at serving time. Its value is as an accuracy reference and as a labeler for generated training data — the focused prompt now labels the `pairing_voyage_gemini` pipeline.

---

## 11. Result 5 — Scoring Fit in Both Directions

### Motivation

Standard similarity scoring asks: does the candidate match the seeker's profile? A professional introduction needs fit in both directions — the candidate's own stated interests should align with what the seeker offers. A great paper-match where only one person wants the other is unlikely to result in an accepted introduction.

This idea is inspired by the Fast-Weight Programmer (FWP) framework from Ga Wu's paper on dynamic reciprocal preference matching. Without user interaction histories, only the static cold-start case is implementable here. **ELABORATION NEEDED** — cite FWP paper.

### Scoring formula

Split each profile into a "looking for" part (the `lookingFor` field plus `searchQuery` for the seeker) and a "background" part (positioning + background fields):

```
forward score = sim(seeker lookingFor, candidate background)
reciprocal score = sim(candidate lookingFor, seeker background)
combined score = forward + λ × reciprocal
```

Both directions use the same frozen embedding model. λ is a single scalar, swept across a grid to find the best value. Retrieval still uses only the forward score; reciprocal scoring only affects the accept/decline prediction. Source: [`docs/reciprocal-static-experiment.md`](reciprocal-static-experiment.md).

### Results across encoders (all 200 pairs)

| Encoder | Forward-only AUC | Combined AUC (best λ) | Held-out combined |
|---|---:|---:|---:|
| Frozen Voyage-4-nano (original fit) | 0.5638 | **0.5964** (λ=1.75, fitted on train) | 0.6241 |
| top1_ctrl fine-tune (λ sweep) | 0.5890 | 0.5961 (λ=0.50) | 0.6853 |
| voyage_gemini_ctrl fine-tune (λ sweep) | 0.6108 | **0.6587** (λ=1.90) | **0.7345** |

On the frozen model, a 5,000-sample bootstrap of the AUC gain gives a 95% confidence interval that includes zero on both the held-out set and all 200 pairs. The direction is consistent, but the size of the gain is not statistically confirmed at this sample size. Source: [`docs/reciprocal-static-experiment.md`](reciprocal-static-experiment.md).

On the `voyage_gemini_ctrl` fine-tune, the all-200 "peak" is a flat shelf across λ=1.0–2.0, not a sharp interior maximum. The held-out peak (λ=0.35, 0.7345) is striking, but this fine-tune was trained on the somewhat-leaky voyage-gemini batch. Source: [`docs/reciprocal-lambda-grid-voyage-gemini-ctrl-experiment.md`](reciprocal-lambda-grid-voyage-gemini-ctrl-experiment.md).

**Consistent finding across all runs:** positive λ helps; negative λ hurts (down to ~0.49 at λ=−2). The reciprocal term carries real directional signal, even if its magnitude is uncertain.

### A worked example

Seeker Adrian is looking for proptech investors. Two candidates are surfaced:

- **Danny (accepted):** a proptech investor looking for founders to back — Danny's `lookingFor` aligns with what Adrian offers (a founder seeking investment)
- **Kunle (declined):** also in proptech, but also raising a fund — both Adrian and Kunle are on the same side of the table

The forward score (does the candidate match the seeker's request?) looks similar for both, since both candidates are proptech-adjacent. The reciprocal score catches the mismatch: Kunle is looking for things Adrian cannot offer.

---

## 12. Result 6 — Score Each Ask Separately

### Motivation

A seeker's `lookingFor` field often covers several distinct goals at once. If the model scores the whole field as one text, a candidate who strongly matches *one* of those goals but misses the others will look the same as a candidate who weakly matches all of them. Treating each paragraph of the wish list as a separate row makes this distinction possible.

### Setup

Each seeker–candidate pair becomes multiple rows, one per paragraph in the seeker's `lookingFor`. A small model (mixture-of-experts with 4 expert networks, 47→24→16 units each) scores each row separately and pools the scores into a pair-level prediction. Inputs are TF-IDF text embeddings of each `lookingFor` paragraph and the candidate's text. The model is trained on 131 real pairs with seeker-disjoint 5-fold cross-validation, repeated across 5 random seeds. Source: [`docs/moe-sectioned-experiment.md`](moe-sectioned-experiment.md).

On average, each seeker has 5.4 `lookingFor` paragraphs, so 131 pairs become 708 scored rows.

### Results (cross-validation, 131 real train pairs)

| Approach | Mean AUC | vs logistic regression |
|---|---:|---|
| Plain logistic regression (pair level) | 0.5758 | — |
| Score each ask, learned attention pooling | **0.6467** | +0.071, wins 5/5 seeds |
| Score each ask, simple average | 0.6404 | +0.065, wins 5/5 seeds |
| Single network (no mixture) | 0.5446 | −0.031 |

The gain over logistic regression (+0.071 AUC, 5/5 seeds) is the most replicated improvement on real training pairs in this project.

**The pooling method does not matter.** Learned attention (which down-weights irrelevant asks) scores almost identically to a plain average (0.6467 vs 0.6404). The gain comes from *scoring each ask separately*, not from cleverly deciding which ask matters most.

**The mixture of experts does matter.** Four expert networks beat one by 0.10 AUC (0.6467 vs 0.5446). Each expert likely specializes on a different type of ask (investors, co-founders, clients, etc.) without being told the ask type.

**ELABORATION NEEDED** — this section reports cross-validation on the train split only; the held-out test set has not been scored by this model, per the decision-gate rule in the fine-tuning plan.

---

## 13. What Did Not Work

Most architectural additions tested in this project did not improve on the simpler alternative. In almost every case, choosing better text to encode produced a larger gain than adding structure to the model.

| Addition | What it was | Result | Simpler alternative that won |
|---|---|---|---|
| Two separate encoders | Train one encoder for the seeker side and a separate one for the candidate side, independently | Below chance on hard-neg AUC (0.483). Two adapters starting from the same base model threw away the base model's alignment between the two sides. | One shared encoder |
| Learned field selector | Encode search query, `lookingFor`, and `positioning` separately; learn a gate to weight them | Worst retrieval of any model tested (R@1 0.23). The gate added 9,000 trainable parameters to a model already prone to overfitting on 643 training rows. | Fixed blend (α=0.6) |
| Drift penalty (KL regularization) | Add a loss term penalizing the fine-tuned model for diverging from the frozen base | Slightly worse on every metric in two separate tests. The collapse it was designed to prevent was not actually happening in the basic training recipe. | Plain training (no penalty) |
| Sharper training objective | Double the loss scaling, focus gradient on the hardest negatives in each batch | All-200 R@1 dropped from 0.18 to 0.14, below the untrained baseline. Sharpening concentrates the gradient on examples the LLM judge gets wrong, which is noise at this label quality. | Default settings |
| Learned score corrections (bilinear MF) | Learn a low-rank correction to the cosine score from frozen embeddings | Inner cross-validation said 0.74 AUC; honest test said 0.62. 131 training pairs is enough to inflate any score with enough hyperparameters. LSA (linear text compression) is the durable finding from this experiment. | Cosine similarity with LSA text compression |
| Forced step-by-step reasoning | Ask the LLM to score six aspects separately, then aggregate | Pair AUC dropped from 0.6409 to 0.6336; yes-rate rose from 55% to 75%. Averaging six middling scores regresses toward the middle. | Naive direct prompt |
| Ten automatic prompt rewrites | Run 20 rounds of LLM-driven prompt revision, scored on real pairs | All ten evolution runs scored below the original prompt. Gentle revision came closest (−0.007 AUC). Stronger seeds, more examples, and larger sampling pools all made it worse. | Naive unmodified prompt |

Sources: [`docs/twotower-split-experiment.md`](twotower-split-experiment.md), [`docs/twotower-field-gate-experiment.md`](twotower-field-gate-experiment.md), [`docs/twotower-kl-reg-experiment.md`](twotower-kl-reg-experiment.md), [`docs/twotower-top1-optimised-experiment.md`](twotower-top1-optimised-experiment.md), [`docs/bilinear-mf-experiment.md`](bilinear-mf-experiment.md), [`docs/llm-judge-experiment.md`](llm-judge-experiment.md), [`docs/judge-prompt-evolution-experiment.md`](judge-prompt-evolution-experiment.md).

---

## 14. Serving

### Why the bi-encoder shape is the only option

The 100 ms response-time budget means the candidate side must be pre-computed. Only the query needs a live pass. This rules out:

- **LLM judge:** each pair takes a separate call, which takes seconds
- **Cross-encoders:** both sides must be encoded together, so every candidate requires a separate forward pass at query time
- **Remote API calls for candidates:** even the Voyage-4-large API likely exceeds the budget for the candidate side at scale

The bi-encoder (separate encoders for each side, dot product to compare) meets the budget: encode candidates offline, store vectors, encode query once at serving time, nearest-neighbor lookup. This is the shape every training run in this project uses.

A merged LoRA adapter adds no serving cost over the frozen base model — the adapter weights are folded into the base weights after training. Fine-tuning is the cheapest way to buy accuracy under this constraint.

### What has not been measured

No latency benchmark exists in this project yet. Every number reported is an offline accuracy number. "Fits the budget" means the architecture is the right shape — whether the actual query-encode plus nearest-neighbor lookup completes in under 100 ms on representative hardware has not been tested. Source: [`docs/objective.md`](objective.md).

The practical open question: can one frozen Voyage-4-nano query encode (the cheapest serving-feasible option) complete in under 100 ms on Boardy's target hardware? That measurement is a prerequisite before calling any configuration a deployment win.

---

## 15. Limitations

### 15.1 The held-out set cannot rank strong models

We withdrew several early claims after re-measuring on all 200 pairs. The 69-pair held-out set has 29 positive queries, so a single extra correct answer changes top-1 hit rate by 3.4 percentage points. Among the top six models, the ranking on the 69-pair set and the ranking on all 200 pairs are uncorrelated (Spearman −0.029). The held-out set screens for clearly broken models but carries no information among working ones — and every decision in this project was made among that top group. Multiple model runs that looked strong on the held-out set reversed completely when measured on all 200 pairs (one showing R@1 0.41 on 69 pairs, R@1 0.13 on all 200). Source: [`docs/all-200-baseline-sweep.md`](all-200-baseline-sweep.md).

### 15.2 Choosing settings on 131 training pairs manufactures results

With 131 training pairs, searching over enough hyperparameter combinations produces a result from selection alone. The bilinear scoring experiment found 0.74 AUC in cross-validation and 0.62 AUC in an honest test — a gap of 0.12 from 64 configurations. The Mixture-of-Experts experiment showed that single-seed runs produced four different rankings across four runs before cross-validation with multiple seeds was introduced. Any experiment reporting a single run on 131 pairs without confidence intervals should be treated as exploratory. Source: [`docs/bilinear-mf-experiment.md`](bilinear-mf-experiment.md), [`docs/moe-sectioned-experiment.md`](moe-sectioned-experiment.md).

### 15.3 Retrieval scores are partly self-graded

The retrieval metrics (top-1 hit rate, MRR) measure whether the model ranks the accepted candidate at the top of a pool of 178 candidates. That pool was selected by Boardy's production system using the same `searchQuery` — so a model that retrieves by query-text similarity partially replicates the production selection. The accept/decline AUC scores do not have this circularity (they measure binary pair classification, not ranking from a production-selected pool). The query-weighted encoding retrieval gains should be read with this in mind. Source: [`docs/query-weighted-encoding-experiment.md`](query-weighted-encoding-experiment.md).

### 15.4 On the larger production export

Boardy shared a larger export (~46,000 rows from the live system). Every frozen model tested on this data lands at or below chance: TF-IDF pair AUC 0.5121, frozen Voyage-4-nano pair AUC 0.4691. Hard-negative AUC inverts (lower AUC means accepted pairs score *lower* similarity than declined ones on average). This is not a contradiction of the 200-pair results — it reflects a different label definition (pending/SMS status rather than a clean accept/decline from a controlled introduction), a much larger candidate pool (21,000+ candidates rather than 178, so retrieval metrics change entirely), and a heavy label skew (~86% accepted when resolved). The export is useful for studying retrieval at scale but not for validating the accept/decline models. Source: [`docs/bdata-tfidf-experiment.md`](bdata-tfidf-experiment.md), [`docs/bdata-voyage-nano-experiment.md`](bdata-voyage-nano-experiment.md).

### 15.5 Generated labels are a model's opinion

The best fine-tuned models were trained on pairs labeled by a Gemini flash-lite or Qwen3-32B LLM. That LLM scores about 0.59–0.64 on the hardest real pairs. Any model trained on those labels cannot exceed what the judge got right. The strongest fine-tune (Qwen, 0.6446) is approaching this ceiling on real pairs, but it was also trained on the leakier `voyage-gemini` batch — so it is unclear how much of the gain is real versus label-structure artifacts. Source: [`docs/moe-rrf003-synthetic-training-findings.md`](moe-rrf003-synthetic-training-findings.md).

---

## 16. Conclusion and Next Steps

### What we learned

The most useful result is a simple one: encode the search query and the profile separately, then blend them. On a frozen model with no training, this nearly doubles the top-1 hit rate and more than doubles the top-10 hit rate. It also turns out to be the representation that fine-tuned models use most effectively — the best nano fine-tune in this project used the query on the seeker side and background+lookingFor on the candidate side, not full profiles on both sides.

The second most useful result is that retrieval and outcome prediction are different problems. The query determines who to show; the pair's structural fit (seniority, stage, two-way interest, timing) determines whether the meeting happens. Models focused on ranking the right candidate often score differently from models focused on accept/decline, and the two do not improve together with the same changes.

Fine-tuning on cleanly generated data does improve accept/decline discrimination, especially on the hard declined pairs — the kind that exist in production. The leakage probes are what makes the difference between a batch that helps and one that hurts: the first 460-pair batch was actively harmful (trained on a batch with 99.2% label-cheatability); subsequent clean batches improved hard-negative AUC by meaningful margins.

Scoring fit in both directions adds signal. Treating each line of the wish list as a separate prediction adds more. Both ideas are consistent across repeated experiments. Neither has been confirmed statistically significant at this sample size, but the direction is clear across all tests.

Eight architectural changes were tried: two separate encoders, a learned field selector, a drift penalty, a sharpened loss, learned score corrections, forced reasoning, and two variations on per-field training. None improved on the simpler alternative of choosing better text to encode. At this data size, complexity is a liability.

### What this points to

The evidence suggests a two-stage design:

1. **Retrieval:** query-weighted bi-encoder search (query vector as the seeker representation) — already the best retrieval in this project (R@1 0.30, R@10 0.91, frozen model, no training)
2. **Re-ranking:** a fast, deployable re-ranker over the top 10–20 candidates that scores fit in both directions or approximates the LLM judge's non-lexical reasoning

The bottleneck for improving step 2 is label quality. An LLM judge scoring ~0.59 on hard pairs is the ceiling for anything trained on its labels. More real accept/decline outcomes from Boardy, or a higher-accuracy labeling model for the hard cases, would unblock further progress more directly than new model architectures.

### Remaining open questions

- Does the reciprocal gain hold up with a proper train→test fit, controlling for the synthetic-batch leakage?
- Can a lightweight re-ranker distill the LLM judge's non-lexical reasoning into something fast enough for the serving budget?
- Does the per-ask sectioned MoE result (cross-validation on 131 train pairs) hold on the 69-pair held-out set? This has deliberately not been checked yet.
- What does the best approach's latency actually look like on Boardy's hardware? No timing measurement has been taken.

---

## Appendix A — Full Results Table

All models on all 200 real pairs, ranked by pair AUC. Fine-tuned models are included here alongside frozen baselines. LLM judges are classification-only (no retrieval metrics).

**ELABORATION NEEDED** — this table will be updated as experiments complete. Numbers may be added for reciprocal-scoring combinations and eval-time query-weighting applied to all fine-tunes.

| Model | Type | Pair AUC | Hard-neg AUC | Easy-neg AUC | MRR | R@1 | R@10 | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| LLM judge, focused prompt | LLM (no retrieval) | **0.6451** | 0.6590 | 0.6570 | — | — | — | [`llm-judge-experiment.md`](llm-judge-experiment.md) |
| LLM judge, naive prompt | LLM (no retrieval) | 0.6177 | — | — | — | — | — | [`llm-judge-experiment.md`](llm-judge-experiment.md) |
| Qwen fine-tune, voyage-gemini batch | Fine-tuned | **0.6446** | **0.6862** | — | 0.415 | 0.24 | — | [`twotower-qwen-voyage-gemini-experiment.md`](twotower-qwen-voyage-gemini-experiment.md) |
| voyage_gemini_ctrl fine-tune + reciprocal (best λ) | Fine-tuned + reciprocal | 0.6587 | — | — | — | — | — | [`reciprocal-lambda-grid-voyage-gemini-ctrl-experiment.md`](reciprocal-lambda-grid-voyage-gemini-ctrl-experiment.md) |
| voyage_gemini_ctrl fine-tune (forward only) | Fine-tuned | 0.6081 | 0.6264 | — | 0.451 | 0.26 | — | [`twotower-voyage-gemini-ctrl-experiment.md`](twotower-voyage-gemini-ctrl-experiment.md) |
| Nano fine-tune, query→bg+lookingFor | Fine-tuned | 0.5983 | **0.6564** | 0.5700 | 0.479 | 0.30 | 0.86 | [`twotower-queryonly-back-look-experiment.md`](twotower-queryonly-back-look-experiment.md) |
| Frozen nano, query-only | Frozen + text choice | 0.5530 | 0.5914 | — | **0.502** | **0.30** | **0.91** | [`query-weighted-encoding-experiment.md`](query-weighted-encoding-experiment.md) |
| Frozen nano, blend α=0.6 | Frozen + text choice | 0.5872 | 0.5818 | — | 0.465 | 0.25 | 0.89 | [`query-weighted-encoding-experiment.md`](query-weighted-encoding-experiment.md) |
| Frozen nano + reciprocal (forward+λ·reverse, λ=1.75) | Frozen + reciprocal | 0.5964 | — | — | — | — | — | [`reciprocal-static-experiment.md`](reciprocal-static-experiment.md) |
| Nano top1_ctrl fine-tune (full text) | Fine-tuned | 0.5683 | 0.5484 | 0.6836 | 0.355 | 0.19 | 0.69 | [`twotower-top1-optimised-experiment.md`](twotower-top1-optimised-experiment.md) |
| Qwen micro-6 fine-tune (rrf_003) | Fine-tuned | 0.5947 | 0.5608 | 0.6708 | 0.303 | 0.14 | 0.66 | [`twotower-qwen-bigbatch-experiment.md`](twotower-qwen-bigbatch-experiment.md) |
| Nano Arm A (real-only) | Fine-tuned | 0.5594 | 0.5558 | 0.6416 | 0.334 | 0.18 | 0.64 | [`twotower-run-001-findings.md`](twotower-run-001-findings.md) |
| **Voyage-4-large (production)** | **Frozen baseline** | **0.5726** | 0.5422 | 0.6540 | 0.310 | 0.13 | 0.70 | [`baseline-results-real200.md`](baseline-results-real200.md) |
| Voyage-4-nano (frozen, full text) | Frozen baseline | 0.5593 | 0.5046 | 0.6960 | 0.317 | 0.18 | 0.59 | [`baseline-results-real200.md`](baseline-results-real200.md) |
| TF-IDF (word overlap) | Lexical baseline | 0.5649 | 0.5164 | 0.6848 | 0.131 | 0.05 | 0.26 | [`baseline-results-real200.md`](baseline-results-real200.md) |
| BGE-en-ICL | Open-weight frozen | 0.5389 | 0.5226 | 0.5928 | 0.319 | 0.17 | 0.62 | [`baseline-results-real200.md`](baseline-results-real200.md) |
| Qwen3-Embedding-8B (frozen) | Open-weight frozen | 0.5529 | 0.4680 | 0.7208 | 0.205 | 0.05 | 0.55 | [`all-200-baseline-sweep.md`](all-200-baseline-sweep.md) |
| E5-Mistral-7B-instruct | Open-weight frozen | 0.4597 | 0.3772 | 0.6144 | 0.116 | 0.03 | 0.30 | [`baseline-results-real200.md`](baseline-results-real200.md) |
| Frozen BERT | Frozen baseline | 0.4697 | 0.4108 | 0.6508 | 0.094 | 0.02 | 0.18 | [`baseline-results-real200.md`](baseline-results-real200.md) |
| NV-Embed-v2 (approx.) | Open-weight frozen | 0.4841 | 0.3836 | 0.6608 | 0.086 | 0.04 | 0.16 | [`baseline-results-real200.md`](baseline-results-real200.md) |
| zembed-1-embedding | Open-weight frozen | 0.4707 | 0.4864 | 0.4816 | 0.038 | 0.01 | 0.08 | [`baseline-results-real200.md`](baseline-results-real200.md) |

*Rows marked with — were not scored on that metric. The Qwen fine-tune on the voyage-gemini batch carries a leakage caveat (Section 15.5). The voyage_gemini_ctrl reciprocal combined AUC is not reported here because the all-200 "peak" is a shelf, not an interior optimum; the 0.6587 figure in the text is the rightmost λ=1.90 value.*

---

## Appendix B — Experiment Ledger

Every experiment run in this project, grouped by track. One row per experiment. **ELABORATION NEEDED** — add run configuration details, training sample sizes, and hardware to each row.

### Track 1: Frozen baselines

| Experiment | What changed | All-200 pair AUC | Verdict | Write-up |
|---|---|---:|---|---|
| Frozen BERT | 512-token BERT-base-uncased | 0.4697 | Floor | [`baseline-metrics.md`](baseline-metrics.md) |
| Voyage-4-nano | Voyage-4-nano, full text | 0.5593 | Baseline | [`baseline-results-real200.md`](baseline-results-real200.md) |
| Voyage-4-large | Boardy production model | 0.5726 | **Production bar** | [`baseline-results-real200.md`](baseline-results-real200.md) |
| TF-IDF | Word-frequency cosine | 0.5649 | Strong on easy-neg, fails retrieval | [`baseline-results-real200.md`](baseline-results-real200.md) |
| BGE-en-ICL | BAAI open-weight, zero-shot | 0.5389 | Best frozen retrieval (MRR 0.319) | [`hf-embedding-baseline-findings.md`](hf-embedding-baseline-findings.md) |
| Qwen3-Embedding-8B (frozen) | 8B open-weight | 0.5529 (retracted 0.6595) | Below production on all-200 | [`all-200-baseline-sweep.md`](all-200-baseline-sweep.md) |
| E5-Mistral-7B-instruct | Instruct-tuned open-weight | 0.4597 | Fails | [`hf-embedding-baseline-findings.md`](hf-embedding-baseline-findings.md) |
| NV-Embed-v2 | NVidia open-weight (approx.) | 0.4841 | Fails; result is approximate | [`hf-embedding-baseline-findings.md`](hf-embedding-baseline-findings.md) |
| zembed-1-embedding | Purpose-built retrieval 4B | 0.4707 | Near-chance, worst retrieval | [`hf-embedding-baseline-findings.md`](hf-embedding-baseline-findings.md) |

### Track 2: Choosing what text to encode

| Experiment | What changed | Key metric | Verdict | Write-up |
|---|---|---|---|---|
| Query ablation (no-query variants) | Remove `searchQuery` from seeker | MRR −19–26% | Query is load-bearing for retrieval | [`baseline-results-real200.md`](baseline-results-real200.md) |
| Field isolation | Embed each field alone, measure identity signal | See Section 7 table | Bottom 3 fields are boilerplate | [`experiment-graphs-index.md`](experiment-graphs-index.md) |
| lookingFor sectioning | Split seeker wish list by paragraph | Pair AUC +0.016 (69-pair) | Seeker-side helps, candidate-side hurts | [`lookingfor-sectioning-findings.md`](lookingfor-sectioning-findings.md) |
| Query-weighted encoding (qw_001) | Separate query + profile vectors, blend α | R@1 0.18→0.30, R@10 0.59→0.91 | **Best retrieval, zero training** | [`query-weighted-encoding-experiment.md`](query-weighted-encoding-experiment.md) |
| Query-weighted on fine-tune (top1_ctrl) | Apply same blend to fine-tuned adapter | R@1 0.19→0.32 | Pattern replicates on fine-tune | [`query-weighted-twotower-experiment.md`](query-weighted-twotower-experiment.md) |
| Voyage-4-large query-only | Query-only on production model | R@1 0.42, MRR 0.590 | **Best retrieval in project (any model)** | [`twotower-no-query-experiment.md`](twotower-no-query-experiment.md) |
| Nomad drift (whole-profile) | Calibrate α on synthetic pairs | Calibration picks α=1.0, real-pair interior at α=0.6 | Boundary-monotonic calibration fails interior | [`nomad-drift-experiment.md`](nomad-drift-experiment.md) |
| Nomad drift (sectioned) | Calibrate section-selected blend | Section wins 4/5 metrics at interior | Section-selected blend is slightly better | [`nomad-drift-sectioned-experiment.md`](nomad-drift-sectioned-experiment.md) |

### Track 3: Fine-tuning

| Experiment | Base model / batch | All-200 pair AUC | Verdict | Write-up |
|---|---|---:|---|---|
| run_001 (ContrastiveLoss, unfixed synth) | Nano / 530-pair (leaky) | 0.578 (69-pair only) | **Failed**: below-chance hard-neg | [`twotower-run-001-results.md`](twotower-run-001-results.md) |
| arm_a_real_only | Nano / 111 real pairs | 0.5594 | Marginal gain; proof synth was harmful | [`twotower-run-001-findings.md`](twotower-run-001-findings.md) |
| distill_judge_001 | Nano / 111 real pairs, soft labels | 0.604 (69-pair) | Soft labels help; checkpoint lost | [`experiment-graphs-index.md`](experiment-graphs-index.md) |
| rrf_triplet_voyage_nano_001 | Nano / rrf_003 triplets | — | First fine-tune to beat large on 69-pair | [`twotower-rrf-triplet-experiment.md`](twotower-rrf-triplet-experiment.md) |
| rrf_triplet_qwen3_8b | Qwen / rrf_003 triplets | — | New 69-pair record at time | [`twotower-rrf-triplet-experiment.md`](twotower-rrf-triplet-experiment.md) |
| abl_a (micro-batch 6, k=1) | Nano / rrf_003 | 0.598 (69-pair) | Best ablation cell; batch size key lever | [`twotower-rrf-triplet-ablation-experiment.md`](twotower-rrf-triplet-ablation-experiment.md) |
| top1_ctrl | Nano / rrf_003, recall@1 checkpoint | 0.5683 | **Best nano (full text)** | [`twotower-top1-optimised-experiment.md`](twotower-top1-optimised-experiment.md) |
| top1_sharp | Nano / rrf_003, sharpened loss | 0.5429 | **Failed**: R@1 below frozen baseline | [`twotower-top1-optimised-experiment.md`](twotower-top1-optimised-experiment.md) |
| no_query_001 | Nano / rrf_003, profile-only seeker at train | 0.5574 | Wash; training text barely matters | [`twotower-no-query-experiment.md`](twotower-no-query-experiment.md) |
| query_only_001 | Nano / rrf_003, query-only seeker at train | 0.5952 | Wash vs eval-time swap | [`twotower-query-only-experiment.md`](twotower-query-only-experiment.md) |
| split_001 | Two separate nano adapters | 0.5677 / hard-neg 0.483 | **Failed**: below-chance hard-neg | [`twotower-split-experiment.md`](twotower-split-experiment.md) |
| field_gate_001 | Learned gate over 3 seeker fields | 0.5919 / worst retrieval | **Failed on retrieval** | [`twotower-field-gate-experiment.md`](twotower-field-gate-experiment.md) |
| kl_reg_ctrl_001 | top1_ctrl + drift penalty | 0.5504 | Slight loss | [`twotower-kl-reg-experiment.md`](twotower-kl-reg-experiment.md) |
| field_bg_look_001 | Two-field text (bg+lookingFor) | 0.5610 | Loses MRR/R@1, wins hard-neg only | [`twotower-field-pairs-experiment.md`](twotower-field-pairs-experiment.md) |
| queryonly_back_look_001 | Query seeker, bg+lookingFor candidate | 0.5983 / hard-neg 0.6564 | **Best nano fine-tune** | [`twotower-queryonly-back-look-experiment.md`](twotower-queryonly-back-look-experiment.md) |
| qwen_micro6 (rrf_003) | Qwen / rrf_003 | 0.5947 | Best rrf_003 pair AUC | [`twotower-qwen-bigbatch-experiment.md`](twotower-qwen-bigbatch-experiment.md) |
| voyage_gemini_ctrl_001 | Nano / voyage-gemini batch | 0.6081 | Beats top1_ctrl (leakier batch) | [`twotower-voyage-gemini-ctrl-experiment.md`](twotower-voyage-gemini-ctrl-experiment.md) |
| voyage_gemini_kl_001 | Nano + drift penalty / voyage-gemini | 0.5479 | **Failed**: KL loses again, wider | [`twotower-voyage-gemini-kl-experiment.md`](twotower-voyage-gemini-kl-experiment.md) |
| qwen_voyage_gemini_001 | Qwen / voyage-gemini batch | 0.6446 / hard-neg 0.6862 | **Project record (accept/decline AUC)** | [`twotower-qwen-voyage-gemini-experiment.md`](twotower-qwen-voyage-gemini-experiment.md) |
| ask_offer_001 | Two towers, Ask+Offer jointly | 0.5714 (combined) | Weak forward-only; reciprocal rescues | [`twotower-ask-offer-experiment.md`](twotower-ask-offer-experiment.md) |
| ask_offer_posbg_001 | Corrected field set for offer | 0.5754 (combined) | Wash vs wide-offer | [`twotower-ask-offer-posbg-experiment.md`](twotower-ask-offer-posbg-experiment.md) |

### Track 4: Generated training data

| Experiment | Scale | Probe results | Verdict | Write-up |
|---|---|---|---|---|
| batch_500_001 (labeled-pair LangGraph) | 460 pairs promoted | Candidate-only AUC 0.992 | **Quarantined**: actively harmful | [`possible-bugs.md`](possible-bugs.md) |
| pair_test_001 (profile-first + fusion labels) | 52 pos / 52 neg | Lexical circularity 0.868 | Labels are query overlap, not fit | [`profile-generation-local-and-bedrock.md`](profile-generation-local-and-bedrock.md) |
| rrf_002 | 275 pairs | Candidate-only 0.634, circ. 0.701 | Passes; seeker base-rate remains | [`rrf-pairing-pipeline.md`](rrf-pairing-pipeline.md) |
| rrf_003 | 2,619 pairs (1,056 triplets) | — | **Training batch for most fine-tunes** | [`rrf-pairing-pipeline.md`](rrf-pairing-pipeline.md) |
| rrf_qwen_full_001 | 25,445 pairs | Seeker-identity 0.739 | Large; train within-seeker only | [`experiment-graphs-index.md`](experiment-graphs-index.md) |
| voyage-gemini batch | ~19,700 pairs | Candidate-only 0.758, seeker-id 0.780 | Leakier; used for Qwen record fine-tune | [`twotower-voyage-gemini-ctrl-experiment.md`](twotower-voyage-gemini-ctrl-experiment.md) |

### Track 5: LLM judge

| Experiment | Variant | All-200 pair AUC | Verdict | Write-up |
|---|---|---:|---|---|
| gemini-3.1-flash-lite, naive | No query | 0.6177 | Best classification (no retrieval) | [`llm-judge-experiment.md`](llm-judge-experiment.md) |
| gemini-3.1-flash-lite, calibrated | Told base rate + production vetting | 0.5901 | Information hurt | [`llm-judge-experiment.md`](llm-judge-experiment.md) |
| gemini-3.1-flash-lite, structured CoT | Six scored aspects | 0.6100 | Regresses to middle | [`llm-judge-experiment.md`](llm-judge-experiment.md) |
| gemini-3.1-flash-lite, focused | Query + trimmed fields | **0.6451** | **Best pair AUC in project** | [`llm-judge-experiment.md`](llm-judge-experiment.md) |
| Gemma-3-27B (Bedrock), naive | No query | 0.5823 | Lower but hard-neg 0.622 | [`llm-judge-experiment.md`](llm-judge-experiment.md) |
| Qwen3-32B (Bedrock), naive | No query | 0.5802 | Lower but hard-neg 0.622 | [`llm-judge-experiment.md`](llm-judge-experiment.md) |
| Qwen3.5-4B (self-hosted), naive+query | With query | 0.5888 | Hard-neg 0.627 (all-200) | [`qwen35-judge-experiment.md`](qwen35-judge-experiment.md) |
| evo_001–evo_009 | Automatic prompt rewrite (9 runs) | 0.5700–0.6105 | **All below naive** | [`judge-prompt-evolution-experiment.md`](judge-prompt-evolution-experiment.md) |
| evo_focused_001 | Evolution on focused seed | 0.5885 vs 0.6474 seed | −0.059 from seed; confidence died | [`judge-prompt-evolution-focused-experiment.md`](judge-prompt-evolution-focused-experiment.md) |

### Track 6: Other model shapes

| Experiment | Approach | Result | Verdict | Write-up |
|---|---|---|---|---|
| MoE reranker (pair-level) | 3-expert MoE over scalars | CV 0.5536 vs 0.5282 no-model | Indistinguishable from chance | [`moe-reranker-experiment.md`](moe-reranker-experiment.md) |
| Sectioned MoE | Per-ask row, 4-expert, 5×5-fold CV | **0.6467 vs 0.5758** logistic | **Best on real train pairs** | [`moe-sectioned-experiment.md`](moe-sectioned-experiment.md) |
| Bilinear MF (LSA arm) | Truncated SVD text compression | Hard-neg AUC +0.037–0.100 | Durable hard-neg improvement | [`bilinear-mf-experiment.md`](bilinear-mf-experiment.md) |
| Bilinear MF (learned score) | Low-rank correction to cosine | Inner CV 0.74, honest 0.62 | Failed: selection overfitting | [`bilinear-mf-experiment.md`](bilinear-mf-experiment.md) |
| Knowledge graph (N=1) | Decompose profiles into typed graphs | Identifies wrong-side mismatch | Motivating diagnostic, not a model | [`knowledge-graph-experiment.md`](knowledge-graph-experiment.md) |
| Reciprocal static (frozen nano) | Forward + λ·reverse scoring | All-200 AUC 0.5964, CI crosses 0 | Directional; not statistically confirmed | [`reciprocal-static-experiment.md`](reciprocal-static-experiment.md) |
| Reciprocal grid (top1_ctrl fine-tune) | Same, on fine-tuned adapter | All-200 AUC 0.5961 | Consistent positive sign | [`reciprocal-lambda-grid-top1ctrl-experiment.md`](reciprocal-lambda-grid-top1ctrl-experiment.md) |
| Reciprocal grid (voyage_gemini_ctrl) | Same, on stronger fine-tune | All-200 0.6587, holdout 0.7345 | Strong but leaky batch caveat | [`reciprocal-lambda-grid-voyage-gemini-ctrl-experiment.md`](reciprocal-lambda-grid-voyage-gemini-ctrl-experiment.md) |

### Track 7: Scale check (B-data)

| Experiment | Model | Pair AUC | Verdict | Write-up |
|---|---|---:|---|---|
| B-data TF-IDF | Word-overlap cosine | 0.5121 | Near chance | [`bdata-tfidf-experiment.md`](bdata-tfidf-experiment.md) |
| B-data Voyage-4-nano | Full profile | 0.4691 | **Below chance** | [`bdata-voyage-nano-experiment.md`](bdata-voyage-nano-experiment.md) |
| B-data Voyage-4-nano (posbg) | Narrow field set | 0.5185 | Near chance; MRR 0.032 at 21k candidates | [`bdata-voyage-nano-posbg-experiment.md`](bdata-voyage-nano-posbg-experiment.md) |
