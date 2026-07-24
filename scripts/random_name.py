"""Programmatic random full-name generation for standalone profile generation.

Exists because the LLM's own fictional-name prior collapses hard: unprompted,
Gemma 3 27B defaults to "Anya Sharma" or "Dr. Aris Thorne" for a large fraction
of generated profiles regardless of archetype (see docs/profile-generation-
local-and-bedrock.md). Generating the name outside the model and handing it in
as a given fact removes the name from the model's own sampled distribution
entirely, rather than trying to steer a distribution that's this skewed.

Not a real-person name list — first/last names are common enough across many
cultures that no single combination should be traceable to one individual, and
combinations are drawn independently (not sourced from real user records).
"""

from __future__ import annotations

import random

FIRST_NAMES = [
    "Aiko", "Aisha", "Alejandro", "Amara", "Amir", "Anh", "Astrid", "Beatriz",
    "Camille", "Chidi", "Daniel", "Diego", "Elena", "Emeka", "Fatima", "Felix",
    "Grace", "Hana", "Hassan", "Ingrid", "Isabela", "Jamal", "Javier", "Jing",
    "Kai", "Kavya", "Kenji", "Layla", "Leilani", "Liang", "Lucas", "Malia",
    "Marcus", "Maria", "Mateus", "Mei", "Mikael", "Nadia", "Naledi", "Nikolai",
    "Noor", "Olamide", "Olivia", "Omar", "Priya", "Rafael", "Ravi", "Rosa",
    "Sana", "Sanjay", "Sofia", "Soraya", "Takumi", "Tariq", "Thabo", "Theo",
    "Valeria", "Viktor", "Wei", "Yara", "Yusuf", "Zainab",
]

LAST_NAMES = [
    "Abara", "Almeida", "Andersson", "Bianchi", "Castillo", "Chen", "Diallo",
    "Dubois", "Eriksson", "Fernandez", "Garcia", "Haddad", "Haruna", "Ibrahim",
    "Ishikawa", "Jansen", "Johansson", "Kaur", "Kimura", "Kowalski", "Lindqvist",
    "Lopes", "Mahmoud", "Martins", "Mendes", "Mensah", "Moreau", "Mueller",
    "Nakamura", "Nguyen", "Novak", "Nkosi", "Oduya", "Okafor", "Olsen",
    "Patel", "Pereira", "Petrov", "Reyes", "Rossi", "Santos", "Silva", "Singh",
    "Sokolov", "Suzuki", "Tanaka", "Torres", "Vargas", "Volkov", "Wang", "Weber",
    "Yamamoto", "Zhang",
]

_rng = random.SystemRandom()  # OS entropy, not a reproducible PRNG seed


def random_full_name() -> str:
    return f"{_rng.choice(FIRST_NAMES)} {_rng.choice(LAST_NAMES)}"
