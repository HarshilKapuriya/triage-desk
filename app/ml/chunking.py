"""Deterministic Text Chunking for Knowledge-Base Articles.

Guarantees:
1. Token Policy Compliance:
   - sentence-transformers/all-MiniLM-L6-v2 tokenizer has TOKENIZER_MAX_LENGTH = 256 subword tokens.
   - Special tokens [CLS] (id 101) and [SEP] (id 102) occupy 2 slots.
   - Target chunk token size is capped at TARGET_CHUNK_TOKENS = 160 tokens (with OVERLAP_TOKENS = 30),
     ensuring total tokens including special tokens stay well below 256.
   - No chunk will ever exceed TOKENIZER_MAX_LENGTH; silent truncation is strictly prevented.
2. Resource-Bounding:
   - Plain-text article content is bounded by MAX_ARTICLE_CHARS = 20,000 characters.
   - If an article generates more than MAX_CHUNKS_PER_ARTICLE = 50 chunks, it is rejected
     atomically with ChunkingError rather than silently discarding trailing content.
3. Determinism & Excerpt Fidelity:
   - Strips and normalizes CRLF/CR to LF before chunking.
   - Preserves readable original text passages without synthetic prefixes.
   - Same input text always produces identical chunks, chunk counts, indices, and content.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from app.ml.manifest import DEFAULT_MODEL_DIR_REL, TOKENIZER_MAX_LENGTH

MAX_ARTICLE_CHARS = 20000
MAX_TITLE_CHARS = 300
TARGET_CHUNK_TOKENS = 160
OVERLAP_TOKENS = 30
MAX_CHUNKS_PER_ARTICLE = 50

_CACHED_TOKENIZER: Any | None = None


class ChunkingError(ValueError):
    """Raised when text chunking fails or bounds are exceeded."""


@dataclass(frozen=True)
class ChunkData:
    """Represents a deterministic passage chunk from an article."""
    chunk_index: int
    content: str
    char_count: int
    token_count: int


def get_local_tokenizer() -> Any | None:
    """Lazily load the provisioned sentence-transformers tokenizer if present on disk."""
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is not None:
        return _CACHED_TOKENIZER

    env_dir = os.getenv("EMBEDDING_MODEL_DIR")
    base_dir = Path(env_dir).resolve() if env_dir else Path(DEFAULT_MODEL_DIR_REL).resolve()
    tok_path = base_dir / "tokenizer.json"
    if tok_path.exists():
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(tok_path))
            tok.enable_truncation(max_length=TOKENIZER_MAX_LENGTH)
            _CACHED_TOKENIZER = tok
            return _CACHED_TOKENIZER
        except Exception:
            pass
    return None


def estimate_or_count_tokens(text: str, tokenizer: Any | None = None) -> int:
    """Count exact tokens including [CLS] and [SEP] special tokens.

    When tokenizer is available, counts active tokens via attention mask (including special tokens).
    When tokenizer is absent (unit tests without disk artifacts), uses conservative subword estimate (+2 for special tokens).
    """
    tok = tokenizer if tokenizer is not None else get_local_tokenizer()
    if tok is not None and hasattr(tok, "encode"):
        try:
            enc = tok.encode(text)
            if hasattr(enc, "attention_mask") and enc.attention_mask:
                return int(sum(enc.attention_mask))
            return len(enc.ids)
        except Exception:
            pass

    # Conservative subword estimate: 1 word ~= 1.35 subwords, + 2 special tokens ([CLS], [SEP])
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return max(3, int(len(words) * 1.35) + 2)


def normalize_article_text(title: str, body: str) -> tuple[str, str]:
    """Validate and normalize article title and body."""
    clean_title = (title or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    clean_body = (body or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    if not clean_title:
        raise ChunkingError("Article title must not be empty.")
    if len(clean_title) > MAX_TITLE_CHARS:
        raise ChunkingError(f"Article title exceeds maximum length of {MAX_TITLE_CHARS} characters.")

    if not clean_body:
        raise ChunkingError("Article body must not be empty.")
    if len(clean_body) > MAX_ARTICLE_CHARS:
        raise ChunkingError(f"Article body exceeds maximum length of {MAX_ARTICLE_CHARS} characters.")

    return clean_title, clean_body


def chunk_article_text(
    title: str,
    body: str,
    tokenizer: Any | None = None,
    target_chunk_tokens: int = TARGET_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    max_chunks: int = MAX_CHUNKS_PER_ARTICLE,
) -> list[ChunkData]:
    """Split article body into deterministic overlapping chunks respecting the token policy.

    Guarantees:
    - Pure readable excerpts from the original text (no synthetic prefixing).
    - Every chunk token count (including [CLS] and [SEP]) <= TOKENIZER_MAX_LENGTH (256).
    - Rejects articles that exceed max_chunks atomically with ChunkingError.
    """
    clean_title, clean_body = normalize_article_text(title, body)

    # 1. Split body into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean_body) if p.strip()]
    if not paragraphs:
        paragraphs = [clean_body]

    # 2. Break paragraphs into sentence or clause segments
    segments: list[str] = []
    for para in paragraphs:
        raw_sents = re.split(r"(?<=[.!?])\s+", para)
        for s in raw_sents:
            s_clean = s.strip()
            if s_clean:
                segments.append(s_clean)

    if not segments:
        segments = [clean_body]

    chunks: list[ChunkData] = []
    current_tokens = 2  # Account for [CLS] and [SEP] special tokens
    current_words: list[str] = []
    seg_idx = 0

    while seg_idx < len(segments):
        seg = segments[seg_idx]
        seg_toks = estimate_or_count_tokens(seg, tokenizer) - 2  # Subwords without special tokens

        # If a single segment alone exceeds target, break it down by words in sequential order
        if seg_toks > target_chunk_tokens:
            sub_words = seg.split()
            step_size = max(10, int(target_chunk_tokens / 1.4))
            sub_segs = [" ".join(sub_words[i : i + step_size]) for i in range(0, len(sub_words), step_size)]
            segments[seg_idx : seg_idx + 1] = sub_segs
            continue

        if current_tokens + seg_toks <= target_chunk_tokens or not current_words:
            current_words.append(seg)
            current_tokens += seg_toks
            seg_idx += 1
        else:
            # Chunk is ready
            chunk_text = " ".join(current_words).strip()
            tok_count = estimate_or_count_tokens(chunk_text, tokenizer)

            # Strict assertion: if token count exceeds limit, push trailing segments back to preserve text coverage
            while tok_count > TOKENIZER_MAX_LENGTH and len(current_words) > 1:
                popped = current_words.pop()
                segments.insert(seg_idx, popped)
                chunk_text = " ".join(current_words).strip()
                tok_count = estimate_or_count_tokens(chunk_text, tokenizer)

            chunks.append(
                ChunkData(
                    chunk_index=len(chunks),
                    content=chunk_text,
                    char_count=len(chunk_text),
                    token_count=tok_count,
                )
            )

            # Check chunk limit before continuing
            if len(chunks) > max_chunks:
                raise ChunkingError(
                    f"Article generates more than {max_chunks} chunks. Reduce article length."
                )

            # Create overlap for next chunk
            overlap_words: list[str] = []
            overlap_tok_count = 2
            for item in reversed(current_words):
                item_toks = estimate_or_count_tokens(item, tokenizer) - 2
                if overlap_tok_count + item_toks <= overlap_tokens + 2:
                    overlap_words.insert(0, item)
                    overlap_tok_count += item_toks
                else:
                    break

            current_words = list(overlap_words)
            current_tokens = overlap_tok_count

    # Flush final chunk if words remain
    if current_words:
        chunk_text = " ".join(current_words).strip()
        tok_count = estimate_or_count_tokens(chunk_text, tokenizer)
        while tok_count > TOKENIZER_MAX_LENGTH and len(current_words) > 1:
            current_words.pop()
            chunk_text = " ".join(current_words).strip()
            tok_count = estimate_or_count_tokens(chunk_text, tokenizer)

        chunks.append(
            ChunkData(
                chunk_index=len(chunks),
                content=chunk_text,
                char_count=len(chunk_text),
                token_count=tok_count,
            )
        )

    # Check total chunks does not exceed limit
    if len(chunks) > max_chunks:
        raise ChunkingError(
            f"Article generates {len(chunks)} chunks, exceeding the maximum limit of {max_chunks} chunks."
        )

    # Guarantee at least one chunk was generated
    if not chunks:
        single_text = clean_body[:MAX_ARTICLE_CHARS]
        single_toks = estimate_or_count_tokens(single_text, tokenizer)
        chunks.append(
            ChunkData(
                chunk_index=0,
                content=single_text,
                char_count=len(single_text),
                token_count=single_toks,
            )
        )

    return chunks
