# TODO: Implement the generator wrapper that produces an answer given a query and retrieved passages.

from __future__ import annotations


def generate(query: str, passages: list[str], max_new_tokens: int) -> str:
    """Generate an answer from a query and retrieved passages."""
    # 1. Format passages into a single block of text
    context = "\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
    
    # 2. Construct the system prompt (Instructions) and the user prompt (System prompt may be unnecessary)
    prompt = f"""
    You are a technical assistant. Answer the user's question using ONLY the provided context.
    If the answer is not in the context, say "I do not have enough information."
    #
    Context:
    {context}
    
    Question: {query}
    Answer:
    """
    
    # 3. Call your LLM (Calling LGMBM)
    response = model(prompt, max_new_tokens)
    return response.text
    
    raise NotImplementedError
