You generate ONE synthetic **hard negative** intro-matching pair for Boardy-style networking.

A hard negative looks topically related (shared industry / keywords) but is a **bad intro**.

Rules:
- Keep the seeker's `userContactFile` and `searchQuery` essentially the same as the seed.
- Invent a NEW match that shares surface jargon with the query but fails for exactly ONE failure mode:
  {failure_mode}
- Failure mode meanings:
  - wrong_side: wrong side of the market (e.g. peer founder when seeker wants investor, or LP↔VC peer)
  - wrong_stage: stage mismatch (pre-seed vs growth, etc.)
  - wrong_role: adjacent domain but wrong role family (advisor vs operator, sales vs peer founder)
  - geo_mismatch: geo/availability conflict when query is geo-specific
  - prefs_conflict: violates clear introPreferences / hard requirements
- Do NOT make an easy unrelated negative (random other industry).
- Do NOT narrate or explain the mismatch anywhere in the candidate's profile.
  Never write meta-commentary that states or hints at the failure_mode itself
  — banned patterns include (but are not limited to) "this would be a
  mismatch," "critical distinction," "mistaken for," "has never," "despite
  the surface similarity," "not a genuine," "not an actual," "important
  distinction for matching," or any sentence whose purpose is to tell the
  reader why this is a bad intro. The candidate must read like a normal,
  self-contained profile — someone who exists and has their own life, not a
  profile written to fail a test. The failure must be inferable only by
  comparing the candidate's facts against the seeker's stated requirements,
  never spelled out in prose.
- Match Boardy CRM tone: markdown-ish lookingFor sections ok; preference
  fields may be empty/null often. Use the same tone, section conventions,
  and level of polish as a genuine positive match profile would have —
  negatives should not be systematically distinguishable from positives by
  writing style, structure, or length alone.
- Use the provided synthetic IDs exactly. Do not copy names/companies/long substrings from few-shots.
- Output a single JSON object with exactly these top-level keys:
  userContactId, matchContactId, userContactFileVersion, matchContactFileVersion,
  searchQuery, userContactFile, matchContactFile
- Each contact file must include: positioning, lookingFor, introPreferences, personalPreferences,
  meetingAndSchedulingPreferences, background, locationAvailability, notes
  (string or null for each nested field).
