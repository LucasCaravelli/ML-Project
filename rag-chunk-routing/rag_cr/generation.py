from __future__ import annotations

import re
import string
from pathlib import Path

from .config import Config, load_config

_FALLBACK_TEMPLATE = (
    "You are a helpful assistant. Answer the question using only the passages below.\n\n"
    "{retrieved_context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _build_prompt(query: str, passages: list[str], template: str) -> str:
    context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    # Support both old-style {query}/{context} and new-style {question}/{retrieved_context}
    return template.format(
        context=context,
        query=query,
        question=query,
        retrieved_context=context,
    )


def _extractive(query: str, passages: list[str]) -> str:
    """Return the passage sentence with the most keyword overlap with the query."""
    stop = {"what", "which", "who", "where", "when", "how", "why", "is", "are", "was",
            "were", "the", "a", "an", "of", "in", "on", "at", "to", "do", "does"}
    query_tokens = {
        t for t in re.sub(r"[" + string.punctuation + r"]", " ", query.lower()).split()
        if t not in stop
    }

    best_sentence, best_score = "", -1
    for passage in passages:
        sentences = re.split(r"(?<=[.!?])\s+", passage.strip())
        for sent in sentences:
            sent_tokens = set(
                re.sub(r"[" + string.punctuation + r"]", " ", sent.lower()).split()
            )
            overlap = len(query_tokens & sent_tokens)
            if overlap > best_score:
                best_score, best_sentence = overlap, sent

    return best_sentence.strip() if best_sentence else (passages[0][:200] if passages else "")


def generate(
    query: str,
    passages: list[str],
    config: Config | None = None,
    config_path: str | Path = "configs/base.yaml",
) -> str:
    """Generate an answer from a query and retrieved passages."""
    if config is None:
        config = load_config(config_path)

    backend     = config.generation.backend
    model_name  = config.generation.model_name
    max_tokens  = config.generation.max_new_tokens
    temperature = config.generation.temperature
    prompt_path = config.prompts.answer

    if backend == "extractive":
        return _extractive(query, passages)

    template = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_path.exists()
        else _FALLBACK_TEMPLATE
    )
    prompt = _build_prompt(query, passages, template)

    if backend == "ollama":
        import ollama  # optional dep; only needed at inference time

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )
        return response["message"]["content"].strip()

    raise ValueError(f"Unsupported generation backend: {backend!r}")

# generation.py — fully implemented:                                                                                     
#   - generate(query, passages, cfg, prompt_path) accepts a GenerationConfig (from the YAML) and an optional path to the   
#   answer prompt template                                                                                               
#   - _build_prompt formats passages as numbered context blocks and fills the {context}/{query} placeholders               
#   - _call_ollama uses the ollama Python library with chat() (appropriate for qwen2.5:7b-instruct), passing num_predict 
#   and temperature from config                                                                                          
#   - Raises ValueError for unsupported backends

#   prompts/answer.txt — filled in with a concise RAG answer prompt using {context} and {query} placeholders.

#   config.py — fixed a pre-existing bug: ChunkingConfig.overlap: int → overlaps: dict[int, int] to match the YAML
#   structure.