You generate ONE synthetic **positive** intro-matching pair for Boardy-style networking.

A positive means: given the seeker's profile + searchQuery, the match profile is a person Boardy should introduce.

Rules:
- Keep the seeker's `userContactFile` and `searchQuery` essentially the same as the seed (light paraphrase of seeker text is ok; do not change intent).
- Invent a NEW match profile that satisfies the query axes: role family, market side, stage, geo (if specified), and lookingFor/intent.
- Use the provided synthetic IDs exactly. Do not copy names, companies, or long substrings from few-shot examples.
- Match Boardy CRM tone: markdown-ish lookingFor sections ok; preference fields may be empty/null often.
- Output a single JSON object with exactly these top-level keys:
  userContactId, matchContactId, userContactFileVersion, matchContactFileVersion,
  searchQuery, userContactFile, matchContactFile
- Each contact file must include: positioning, lookingFor, introPreferences, personalPreferences,
  meetingAndSchedulingPreferences, background, locationAvailability, notes
  (string or null for each nested field).
