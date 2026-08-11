You will be shown a judge prompt that has grown over several rounds of
editing. Distill it into a shorter version that keeps the same underlying
decision-making ability but expresses it as general principles, not a list
of specific rules or cases.

Look for rules that are really the same idea applied to different
situations, and merge them into one statement of the idea. Cut anything that
restates the same point in different words. Prefer one well-chosen sentence
of judgment over three sentences of examples. The result should read like
someone who deeply understands the underlying pattern, not someone listing
every case they've seen.

Do not lose real distinctions the prompt has learned — only remove
redundancy and over-specification. If two rules genuinely cover different
situations, keep both, just state them briefly.

Hard constraints — the distilled prompt MUST still:
- Output a single JSON object with exactly these keys: `reasoning` (2-4
  sentences, written before deciding), `match` ("yes"/"no"), `confidence`
  (integer 0-100). No other keys, no prose or markdown around it.
- Frame the task as: two complete profiles, Person A and Person B, would
  introducing them be a good match?
- Stay naive: never mention that production already filtered these pairs,
  never state a base rate, never reference this evaluation setup.

Respond with a single JSON object and nothing else:

{
  "updated_prompt": "<the full distilled prompt, ready to use as-is>",
  "rationale": "<1-3 sentences: what you merged or cut, and why>"
}
