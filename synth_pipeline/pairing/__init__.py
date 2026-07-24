"""Turn standalone unlabeled profiles into labeled pos/neg pairs.

This is the step after `scripts/bedrock_profile_gen.py`. Profiles arrive with no
label, no counterpart, and no pair context — the whole point of the profile-first
redesign (see `docs/possible-bugs.md` #4, where generating a pair *given* a label
let the LLM leak stylistic tells of that label into the profile text).

Stages, in order:
    profiles.py  load a generation run, mint contact IDs, drop `reasoning`
    query.py     synthesize a searchQuery per profile (the only LLM call here)
    select.py    rank candidates against each query, pick a top band
    label.py     score with the TF-IDF + Voyage-nano fusion, label with a deadband
    stage.py     write batch-isolated envelopes under artifacts/pairing/<batch_id>/

Labels come from the hybrid scorer, not an LLM judge. That makes them provisional:
a model trained on this data can at best imitate the scorer that labeled it. Batches
stay in their own namespace and are never promoted into data/dataset_*.json.
"""
