# TODO: Implement the generator wrapper that produces an answer given a query and retrieved passages.

from __future__ import annotations
import yaml
import ollama
from pathlib import Path

def load_config(config_path: str = "../configs/base.yaml") -> dict:
    """Loads the YAML configuration file."""
    # Using Path to ensure we find the file relative to this script
    base_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    with open(base_path, "r") as f:
        return yaml.safe_load(f)

def generate(query: str, passages: list[str]) -> str:
    """
    Generate an answer using hyperparameters referenced from base.yaml.
    """
    # 1. Load configuration
    config = load_config()
    gen_cfg = config.get("generation", {})
    
    # 2. Format passages
    context = "\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
    
    # 3. Construct the prompt
    prompt = f"""You are a technical assistant. Answer the user's question using ONLY the provided context.
If the answer is not in the context, say "I do not have enough information."

Context:
{context}

Question: {query}
Answer:"""

    # 4. Call Ollama using values from the YAML
    # We use .get() to provide fallbacks just in case the YAML is missing a key
    response = ollama.generate(
        model=gen_cfg.get("model_name", "qwen2.5:7b-instruct"),
        prompt=prompt,
        options={
            'num_predict': gen_cfg.get("max_new_tokens", 256),
            'temperature': gen_cfg.get("temperature", 0.0),
        }
    )
    
    return response['response'].strip()
