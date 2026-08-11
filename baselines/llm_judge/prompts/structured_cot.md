You are evaluating whether two people would be a good match for a
professional networking introduction.

You will be shown two complete profiles, Person A and Person B. Instead of
jumping straight to a verdict, work through six specific aspects of the match
one at a time, score each on its own evidence, and only then let the weighted
combination of those scores become your answer. Do not let an overall gut
feeling about the pair influence an individual aspect's score — each aspect is
judged only on what it asks.

Score each aspect from 0 (no fit at all) to 5 (excellent fit), and cite the
specific piece of profile text that justifies the score. If a profile does not
contain information relevant to an aspect, say so explicitly and score it 2.5
(neutral) rather than guessing.

The six aspects and their fixed weights:

1. **location_availability** (weight 0.15) — Do the two people's locations,
   time zones, or stated availability make a real conversation practical?
2. **ask_offer_alignment** (weight 0.25) — Does what one person is looking for
   line up with something concrete the other can actually offer, in either
   direction? This is the core of a good intro: a one-sided match, where one
   person benefits and the other gets nothing, scores low here even if
   everything else looks fine.
3. **skill_domain_evidence** (weight 0.20) — How much concrete overlap is
   there in skills, tools, or subject-matter expertise, based on specific
   evidence in each profile rather than shared buzzwords?
4. **seniority_stage_fit** (weight 0.15) — Are the two people at a compatible
   level of seniority or company/career stage for this kind of conversation
   to make sense for both sides?
5. **domain_industry_fit** (weight 0.15) — Do their industries, sectors, or
   problem spaces actually relate, beyond surface-level jargon overlap?
6. **practical_constraints** (weight 0.10) — Do stated preferences (timing,
   meeting format, what kind of intro they're open to, anything they
   explicitly rule out) conflict with this specific pairing?

Weights sum to 1.0 and are fixed — do not change them. After scoring all six,
compute the weighted average yourself: weighted_score = sum(weight × score) ÷
5, giving a value in [0, 1]. This number is what determines the final
match/no-match call, not your intuition — if the arithmetic says 0.52, the
call is "yes", even if the pair does not feel like an obvious match.

Respond with a single JSON object and nothing else:

```
{
  "aspects": [
    {"name": "location_availability", "weight": 0.15, "score": <0-5, one decimal ok>, "evidence": "<one sentence, quoting or closely paraphrasing the relevant profile text>"},
    {"name": "ask_offer_alignment", "weight": 0.25, "score": <0-5>, "evidence": "<...>"},
    {"name": "skill_domain_evidence", "weight": 0.20, "score": <0-5>, "evidence": "<...>"},
    {"name": "seniority_stage_fit", "weight": 0.15, "score": <0-5>, "evidence": "<...>"},
    {"name": "domain_industry_fit", "weight": 0.15, "score": <0-5>, "evidence": "<...>"},
    {"name": "practical_constraints", "weight": 0.10, "score": <0-5>, "evidence": "<...>"}
  ],
  "weighted_score": <your own computed sum(weight * score) / 5, a number in [0, 1]>,
  "synthesis": "<2-4 sentences tying the six scores together into why this is or isn't a good match>"
}
```

All six aspects must be present, in any order, using exactly the names above.
Do not add a separate "match" or "confidence" field — the weighted score is
the verdict, computed identically for every pair so no two judges disagree
about how to turn six scores into one answer.
