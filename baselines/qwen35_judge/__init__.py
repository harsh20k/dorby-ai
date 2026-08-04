"""LLM-judge experiment using a self-hosted, open-weight Qwen3.5-4B on Modal.

Same evaluation shape as ``baselines/llm_judge`` (prompted, no fine-tuning
yet — that's step two) but a different backend: instead of an OpenRouter/
Bedrock API call, the model runs on a Modal GPU from Hugging Face weights.
Prompt is ``naive`` from ``baselines/llm_judge/prompt.py`` plus
``searchQuery`` (the one deliberate difference — see ``prompts/naive_query.md``
and ``docs/qwen35-judge-experiment.md``), pulled from LangSmith Hub so the
exact prompt behind any run's numbers is a named, inspectable commit.
"""
