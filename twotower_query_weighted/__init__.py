"""Query-weighted seeker encoding, applied to the fine-tuned two-tower model.

``query_weighted/`` found that on the *frozen* voyage-4-nano encoder, blending
the query and profile as separate vectors — ``normalize(alpha*query + (1-alpha)*profile)``
— roughly doubles recall@1 over the baseline's simple text concatenation. This
package asks whether that holds for the *fine-tuned* encoder too, using the
best fine-tune in the project so far
(``artifacts/twotower_top1_optimised/top1_ctrl_001``, all-200 recall@1 0.19,
see ``docs/twotower-top1-optimised-experiment.md``).

Isolated from both ``query_weighted/`` and ``twotower/`` per the repo's
experiment-isolation rule: no training happens here (the adapter is loaded
read-only, exactly as ``eval_real_full/eval.py`` already does), and no file in
either prior package is edited. Only ``twotower.eval.load_model_for_eval`` /
``encode_role`` and ``baselines.metrics`` are imported, unchanged, as public
API.
"""
