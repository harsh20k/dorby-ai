"""Matrix factorization on text, in the two places it can actually apply here.

Recommender-systems matrix factorization factors a ``users x items`` interaction
matrix into two thin matrices of learned per-user and per-item vectors. That
shape does not transfer to this project directly: the interaction matrix is 129
seekers x 178 candidates with only 200 filled cells, and a freely-learned vector
per contact would be both unidentifiable at that density and useless for anyone
unseen. Every deployable model here has to read *text*.

So this package tests the two places factorization survives that constraint:

``lsa``
    Classic **text** matrix factorization — truncated SVD of the TF-IDF
    ``documents x terms`` matrix (LSA/LSI), then cosine in the compressed space.
    Label-free, so it is a fair drop-in alternative to the frozen encoders and
    can be scored on any subset without leakage.

``bilinear``
    Factorization of the **scoring function** rather than of the text. Keeps a
    frozen encoder's vectors and learns a low-rank correction on top:

        score(s, c) = cos(s, c) + (A s) . (B c),   A, B in R^{k x d}

    which is exactly ``s^T W c`` with ``W = I + A^T B`` constrained to rank
    ``k`` off the identity. This is the content-based form of MF: the per-user
    and per-item vectors are *computed from text* by ``A`` and ``B`` instead of
    looked up, so it generalizes to unseen contacts. It also buys the one thing
    plain cosine structurally cannot express — an asymmetric, non-identity
    metric coupling different directions of seeker space to candidate space.

Both arms are scored through ``baselines.metrics`` unchanged, so their numbers
are directly comparable to every row in ``docs/baseline-results-real200.md``.

Isolation: nothing under ``baselines/``, ``eval_real_full/`` or ``twotower/`` is
modified. Read-only imports only, per the rule in CLAUDE.md.
"""
