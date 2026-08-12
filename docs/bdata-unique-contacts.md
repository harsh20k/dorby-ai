# B-data unique contacts

Catalog of unique people inferred from locked `data/B-data.json`.
Candidates have no `contactId`, so identity is a hash of `positioning`
(fallback: first nonempty other profile field).

## Result

**29,923 unique contacts** (9 all-empty profiles dropped).

| Slice | n |
|---|---:|
| Keyed on positioning | 29,904 |
| Fallback background | 11 |
| Fallback lookingFor | 8 |
| Role: both seeker + candidate | 10,333 |
| Role: seeker only | 8,755 |
| Role: candidate only (no id) | 10,835 |

Published:
[https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-unique-contacts.html](https://dorby-project-story-411960113601.s3.amazonaws.com/docs/bdata-unique-contacts.html)

Local: `docs/html/bdata-unique-contacts.html` ·
JSON: `data/unique_contacts_B_data.json` (gitignored) ·
stats: `artifacts/bdata_unique_contacts/stats.json`

## Identity rule

1. Nonempty `positioning` → `sha256("positioning\\0" + stripped text)`
2. Else first nonempty of `background`, `lookingFor`, `locationAvailability`,
   `introPreferences`, `personalPreferences`, `meetingAndSchedulingPreferences`
3. All-empty → dropped
4. Collisions keep the richest `contactFile`; union seeker ids and queries

Inferred, not Boardy ids: two different positioning strings → two people;
identical positioning → one.

## Reproduce

```bash
python scripts/build_unique_contacts_B_data.py
python -m pytest tests/test_unique_contacts_B_data.py -q
```

Does not write to `data/B-data.json`.
