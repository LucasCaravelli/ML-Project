"""Answer generation backends (vLLM / ollama) behind a single ``generate`` interface."""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Any

from .config import Config, load_config

_FALLBACK_TEMPLATE = (
    "You are a helpful assistant. Answer the question using only the passages below.\n\n"
    "{retrieved_context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

_VLLM_CACHE: dict[str, Any] = {}


def _build_prompt(query: str, passages: list[str], template: str) -> str:
    context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    # Support both old-style {query}/{context} and new-style {question}/{retrieved_context}
    return template.format(
        context=context,
        query=query,
        question=query,
        retrieved_context=context,
    )


def _load_template(cfg: Config) -> str:
    prompt_path = cfg.prompts.answer
    return prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else _FALLBACK_TEMPLATE


def build_prompt(query: str, passages: list[str], cfg: Config) -> str:
    """Public helper so callers can build prompts before batching."""
    return _build_prompt(query, passages, _load_template(cfg))


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


def _get_vllm(model_name: str) -> Any:
    if model_name not in _VLLM_CACHE:
        from vllm import LLM  # imported lazily so local dev doesn't need vllm

        _VLLM_CACHE[model_name] = LLM(
            model=model_name,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.85,
        )
    return _VLLM_CACHE[model_name]


def _generate_vllm(prompts: list[str], cfg: Config) -> list[str]:
    from vllm import SamplingParams

    llm = _get_vllm(cfg.generation.model_name)
    params = SamplingParams(
        temperature=cfg.generation.temperature,
        max_tokens=cfg.generation.max_new_tokens,
    )
    # Qwen2.5-Instruct expects the chat template (<|im_start|>...). Calling
    # llm.generate() with raw text bypasses instruction tuning and degrades
    # answer quality; llm.chat() applies the tokenizer's chat template.
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(conversations, params)
    return [o.outputs[0].text.strip() for o in outputs]


def _generate_ollama(prompts: list[str], cfg: Config) -> list[str]:
    import ollama

    results: list[str] = []
    for prompt in prompts:
        response = ollama.chat(
            model=cfg.generation.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": cfg.generation.max_new_tokens,
                "temperature": cfg.generation.temperature,
            },
        )
        results.append(response["message"]["content"].strip())
    return results


def generate_batch(prompts: list[str], cfg: Config) -> list[str]:
    """Generate answers for many prompts in one pass.

    vLLM batches with continuous decoding; Ollama loops one-at-a-time;
    the extractive backend takes no prompts (it'll error here — use
    `generate()` instead for that backend since it doesn't need a prompt
    template).
    """
    if not prompts:
        return []
    backend = cfg.generation.backend
    if backend == "vllm":
        return _generate_vllm(prompts, cfg)
    if backend == "ollama":
        return _generate_ollama(prompts, cfg)
    raise ValueError(f"generate_batch does not support backend {backend!r}")


def generate(
    query: str,
    passages: list[str],
    config: Config | None = None,
    config_path: str | Path = "configs/base.yaml",
) -> str:
    """Generate an answer from a query and retrieved passages."""
    if config is None:
        config = load_config(config_path)

    if config.generation.backend == "extractive":
        return _extractive(query, passages)

    prompt = build_prompt(query, passages, config)
    return generate_batch([prompt], config)[0]
