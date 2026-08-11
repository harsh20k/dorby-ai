#!/usr/bin/env python3
"""CLI entrypoint for the judge-prompt-evolution experiment.

    python scripts/run_judge_prompt_evolution.py
    python scripts/run_judge_prompt_evolution.py --n-iterations 20 --run-id rrf_judge_opt_001
    python scripts/run_judge_prompt_evolution.py --no-hub   # local files only

See judge_prompt_evolution/ for the isolated package this wraps.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judge_prompt_evolution.run import main

if __name__ == "__main__":
    raise SystemExit(main())
