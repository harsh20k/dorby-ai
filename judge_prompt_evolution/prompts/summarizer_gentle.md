You will be shown a judge prompt that has grown over several rounds of
editing. Revise it so its ideas are stated more clearly, as general
principles rather than a list of narrow special cases — but do not aim for
brevity as a goal in itself. Length is not the problem being solved here;
clarity and generalization are.

Where two or more rules are really the same idea applied to different
situations, merge them into one clearly-stated principle — but keep whatever
detail is needed for that principle to actually be usable, rather than
collapsing it into a single vague sentence. If a rule captures a real,
specific distinction that a shorter phrasing would lose, keep it close to
its current length. Only cut wording that is purely repetitive — the same
point stated twice in different words — never a distinct case or condition,
even a narrow one.

When in doubt, prefer keeping a rule at its current length and clarity over
shortening it. A good outcome here may end up close to the same size as what
you started with; that is fine. The goal is a clearer rubric, not a shorter
one.

Hard constraints — the revised prompt MUST still:
- Output a single JSON object with exactly these keys: `reasoning` (2-4
  sentences, written before deciding), `match` ("yes"/"no"), `confidence`
  (integer 0-100). No other keys, no prose or markdown around it.
- Frame the task as: two complete profiles, Person A and Person B, would
  introducing them be a good match?
- Stay naive: never mention that production already filtered these pairs,
  never state a base rate, never reference this evaluation setup.

Respond with a single JSON object and nothing else:

{
  "updated_prompt": "<the full revised prompt, ready to use as-is>",
  "rationale": "<1-3 sentences: what you clarified or merged, and why>"
}
