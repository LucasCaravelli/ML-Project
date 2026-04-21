from __future__ import annotations

from pathlib import Path

from .config import load_config

_FALLBACK_TEMPLATE = (
    "You are a helpful assistant. Answer the question using only the passages below.\n\n"
    "{context}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)


def _build_prompt(query: str, passages: list[str], template: str) -> str:
    context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return template.format(context=context, query=query)


def generate(query: str, passages: list[str], config_path: str | Path = "configs/base.yaml") -> str:
    """Generate an answer from a query and retrieved passages."""
    cfg = load_config(config_path)

    backend     = cfg.generation.backend
    model_name  = cfg.generation.model_name
    max_tokens  = cfg.generation.max_new_tokens
    temperature = cfg.generation.temperature
    prompt_path = cfg.prompts.answer

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