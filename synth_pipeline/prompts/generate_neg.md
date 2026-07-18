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
- Use the provided synthetic IDs exactly. Do not copy names/companies/long substrings from few-shots.
- Output a single JSON object with exactly these top-level keys:
  userContactId, matchContactId, userContactFileVersion, matchContactFileVersion,
  searchQuery, userContactFile, matchContactFile
- Each contact file must include: positioning, lookingFor, introPreferences, personalPreferences,
  meetingAndSchedulingPreferences, background, locationAvailability, notes
  (string or null for each nested field).
