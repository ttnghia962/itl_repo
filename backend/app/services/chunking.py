from typing import List


def chunk_text(text: str, max_tokens: int = 200, overlap_tokens: int = 20) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_tokens:
        return [text.strip()]
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    chunks: List[str] = []
    step = max_tokens - overlap_tokens
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks
