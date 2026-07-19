# Possible bugs

Running log of suspected pipeline/data issues found during review, not yet
root-caused or fixed. Add an entry when something looks wrong but hasn't
been confirmed/patched; move it out (or mark fixed) once resolved.

## 1. Long seeker `lookingFor` gets truncated inconsistently with `searchQuery` in negative generation

**Where:** `synth_pipeline/nodes/generate.py` (negative-generation path),
prompt `synth_pipeline/prompts/generate_neg.md`.

**Found in:** `artifacts/synth/batch_500_001/staged/neg_cmsynthtcfm6yqmmq5rxb5lif.json`
(seeker `cmsynth2gb2fna51xgpcxuu1w`, seeded from real pair
`neg:cmohrqn4800kimu02s837gmav:...` in `data/dataset_negative.json`).

**What's wrong:** The prompt instructs the model to "keep the seeker's
`userContactFile` and `searchQuery` essentially the same as the seed." The
`searchQuery` was copied verbatim from the seed ("hands-on science
researchers founders or CSOs building biotech and deep-tech companies...").
But the seed's `lookingFor` field is a sprawling ~15-section field
(accumulated over many real update timestamps) covering real estate/car
wash funding, growth roles, *and* a "Science-First Biotech Founders and
CSOs" section that the query is drawn from almost word-for-word.

The generated pair's `userContactFile.lookingFor` kept only 4 of the ~15
sections (Commercial Real Estate, Growth Roles, Car Wash Funding, Real
Estate Capital) and dropped the biotech section — the one section that
actually justifies the `searchQuery`. Result: a seeker profile with no
stated interest in biotech, paired with a biotech-seeking query.

**Why it matters:** The judge still labeled the pair a valid negative, but
for the wrong reason — its own `judge_raw.reason` says "zero alignment
between the user's search query... and the user's actual professional
focus," failing role/side/stage/prefs axes. The pair was tagged
`failure_mode: wrong_stage` (Eleanor's Series B stage vs. Max's early-stage
ask), but the actual defect is seeker-query inconsistency introduced during
generation, not a genuine stage mismatch on the match side. Training on
this teaches "reject when query doesn't match profile" (an artifact of
generation) rather than the intended failure-mode semantics.

**Suspected root cause:** very long/multi-section `lookingFor` fields
likely push the generation LLM toward summarizing/compressing rather than
copying verbatim, since the instruction is qualitative ("essentially the
same") with no explicit length/fidelity constraint.

**Suggested next step:** grep `data/dataset_negative.json` /
`dataset_positive.json` for seed users with unusually long `lookingFor`
strings, cross-reference which batch_500_001 pairs were generated from
those seeds, and spot-check whether the same section-dropping pattern shows
up elsewhere. If confirmed widespread, consider passing `lookingFor`
through unmodified (string substitution rather than LLM regeneration) for
the seeker side, since the prompt already says it should stay unchanged.

**Status:** unconfirmed at scale — one instance found via manual review
(2% sample). Human-approved as staged despite the defect (2026-07-19); flag
for exclusion or fix before scaling generation further.
