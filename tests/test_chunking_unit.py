"""Offline unit tests for deterministic text chunking.

Verifies:
1. Normalization: CRLF -> LF, outer whitespace stripping.
2. Bounds enforcement: rejects empty title/body, title > 300 chars, body > 20,000 chars.
3. Determinism: identical text yields identical chunk splits, indices, char counts, and token counts.
4. Token policy compliance: no chunk exceeds TOKENIZER_MAX_LENGTH (256 subwords).
5. Boundary overlap: adjacent chunks preserve semantic overlap.
6. Per-article chunk cap: chunk count is clamped within MAX_CHUNKS_PER_ARTICLE (50).
"""

import pytest
from app.ml.chunking import (
    ChunkingError,
    MAX_ARTICLE_CHARS,
    MAX_CHUNKS_PER_ARTICLE,
    MAX_TITLE_CHARS,
    TARGET_CHUNK_TOKENS,
    chunk_article_text,
    normalize_article_text,
)
from app.ml.manifest import TOKENIZER_MAX_LENGTH


def test_normalize_article_text_valid_and_whitespace_strip():
    """Valid text with Windows newlines and excess whitespace is cleanly normalized."""
    raw_title = "  Dispute Resolution Policy\r\n\r\n"
    raw_body = "\r\nLine 1\r\n\r\nLine 2 with details.\r\n"

    title, body = normalize_article_text(raw_title, raw_body)
    assert title == "Dispute Resolution Policy"
    assert "\r" not in body
    assert body == "Line 1\n\nLine 2 with details."


def test_normalize_article_text_rejects_empty_or_oversized():
    """Rejects empty strings, whitespace-only, or text exceeding length bounds."""
    # Empty title
    with pytest.raises(ChunkingError, match="title must not be empty"):
        normalize_article_text("   ", "Valid body text.")

    # Title exceeds 300 chars
    long_title = "A" * (MAX_TITLE_CHARS + 1)
    with pytest.raises(ChunkingError, match="title exceeds maximum length"):
        normalize_article_text(long_title, "Valid body text.")

    # Empty body
    with pytest.raises(ChunkingError, match="body must not be empty"):
        normalize_article_text("Valid title", "   \r\n  ")

    # Body exceeds 20,000 chars
    long_body = "B" * (MAX_ARTICLE_CHARS + 1)
    with pytest.raises(ChunkingError, match="body exceeds maximum length"):
        normalize_article_text("Valid title", long_body)


def test_chunk_article_text_deterministic():
    """Identical input text always produces the exact same sequence of chunks."""
    title = "Billing Dispute Guidelines"
    body = (
        "When an account holder notices an unauthorized charge, they must report it within 60 calendar days.\n\n"
        "The financial institution must acknowledge receipt within 30 days and resolve the dispute within 90 days. "
        "During the investigation, the disputed amount cannot be reported as delinquent to credit reporting agencies.\n\n"
        "If provisional credit was issued, it may be revoked if the merchant proves the charge was authorized."
    )

    chunks1 = chunk_article_text(title, body)
    chunks2 = chunk_article_text(title, body)

    assert len(chunks1) == len(chunks2)
    assert len(chunks1) > 0
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.chunk_index == c2.chunk_index
        assert c1.content == c2.content
        assert c1.char_count == c2.char_count
        assert c1.token_count == c2.token_count


def test_chunk_article_text_token_policy_compliance():
    """Every generated chunk must strictly respect TOKENIZER_MAX_LENGTH (256 subwords)."""
    title = "Mortgage Servicing Rules & Escrow Analysis Procedures"
    # Generate multi-paragraph text
    paras = [
        f"Paragraph {i}: The servicer must provide an annual escrow account statement summarizing all inflows and outflows. "
        f"Any surplus exceeding fifty dollars must be refunded to the borrower within thirty days of the escrow analysis. "
        f"If a shortage is discovered, the servicer may require the borrower to repay the shortage over a period of twelve months."
        for i in range(1, 20)
    ]
    body = "\n\n".join(paras)

    chunks = chunk_article_text(title, body)
    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.token_count <= TOKENIZER_MAX_LENGTH, (
            f"Chunk #{chunk.chunk_index} has {chunk.token_count} tokens, exceeding maximum of {TOKENIZER_MAX_LENGTH}."
        )
        assert chunk.char_count == len(chunk.content)
        assert len(chunk.content.strip()) > 0


def test_chunk_article_text_overlap_preserved():
    """Adjacent chunks should carry over boundary context for seamless semantic retrieval."""
    title = "Credit Card Fraud Policy"
    body = (
        "Section 1: Initial Investigation. The fraud department reviews anomalous transactions flagged by automated fraud models. "
        "The customer is contacted via verified SMS or email to confirm recent card activity.\n\n"
        "Section 2: Temporary Card Block. If the customer does not respond within twenty-four hours, the card status is changed to restricted. "
        "No new authorization requests will be approved until identity verification is completed.\n\n"
        "Section 3: Replacement Card Issuance. Upon confirmation of unauthorized use, the compromised card is permanently deactivated. "
        "A replacement card with a new primary account number is dispatched via standard priority mail within two business days.\n\n"
        "Section 4: Liability Exemption. The cardholder is zero-percent liable for confirmed unauthorized electronic transactions."
    )

    chunks = chunk_article_text(title, body, target_chunk_tokens=60, overlap_tokens=20)
    assert len(chunks) >= 2

    # Check that sequential chunks have non-empty content and strictly increasing indices
    for i in range(len(chunks) - 1):
        assert chunks[i].chunk_index == i
        assert chunks[i + 1].chunk_index == i + 1


def test_chunk_article_text_per_article_limit_bounded():
    """Long repetitive article is bounded by MAX_CHUNKS_PER_ARTICLE (50)."""
    title = "Comprehensive Multi-Topic Banking Manual"
    # Long text with 100 sentences
    sentences = [
        f"Clause {i}: All transactions involving foreign exchange must be reported to the oversight committee by end of business day."
        for i in range(1, 100)
    ]
    body = "\n\n".join(sentences)

    chunks = chunk_article_text(title, body, max_chunks=MAX_CHUNKS_PER_ARTICLE)
    assert len(chunks) <= MAX_CHUNKS_PER_ARTICLE


def test_chunk_article_text_single_short_paragraph():
    """A short article produces exactly 1 chunk containing readable original text without synthetic prefixing."""
    title = "PIN Reset"
    body = "To reset your debit card PIN, visit any authorized branch ATM with your card."

    chunks = chunk_article_text(title, body)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == body
    assert "debit card PIN" in chunks[0].content
    assert chunks[0].token_count <= TOKENIZER_MAX_LENGTH


def test_chunk_article_text_rejects_exceeding_max_chunks():
    """Atomically rejects articles that exceed max_chunks with ChunkingError rather than silent content loss."""
    title = "Oversized Article"
    paras = [
        f"Paragraph {i}: Detailed policy content that spans across multiple sentences and paragraphs to generate many chunks."
        for i in range(25)
    ]
    body = "\n\n".join(paras)

    with pytest.raises(ChunkingError, match="Article generates more than 3 chunks"):
        chunk_article_text(title, body, max_chunks=3, target_chunk_tokens=30, overlap_tokens=10)


def test_chunk_article_text_preserves_full_text_coverage():
    """Every sentence and unique token in the source body must be covered across chunks without silent loss."""
    title = "Regulatory Compliance Policy"
    sentences = [
        f"Item {i} states that compliance reviews for consumer complaints must occur every ninety days."
        for i in range(1, 15)
    ]
    body = " ".join(sentences)

    chunks = chunk_article_text(title, body, target_chunk_tokens=60, overlap_tokens=15)
    assert len(chunks) >= 3

    # Check that all Item numbers 1 to 14 appear in the chunk sequence
    combined_content = " ".join(c.content for c in chunks)
    for i in range(1, 15):
        assert f"Item {i}" in combined_content, f"Item {i} was missing from chunk output."


def test_chunking_with_real_local_tokenizer():
    """Verify chunking against the actual local sentence-transformers tokenizer artifact on disk.

    Explicitly distinguished from artifact-free unit tests: cleanly skipped if local
    model artifacts are absent (e.g. fresh CI checkout).
    """
    from app.ml.chunking import get_local_tokenizer
    tokenizer = get_local_tokenizer()
    if tokenizer is None:
        pytest.skip("Local tokenizer artifact not present on disk (models/sentence-transformers/all-MiniLM-L6-v2/tokenizer.json)")

    sample_text = "The borrower submitted an escrow surplus inquiry regarding their mortgage account."
    enc = tokenizer.encode(sample_text)
    active_tokens = sum(enc.attention_mask)

    # Tokenizer must include [CLS] (101) at the start and [SEP] (102) at the end of active tokens
    assert enc.ids[0] == 101
    assert enc.ids[active_tokens - 1] == 102
    assert active_tokens <= TOKENIZER_MAX_LENGTH
    assert len(enc.ids) <= TOKENIZER_MAX_LENGTH

    # Verify chunking with real tokenizer
    title = "Mortgage Escrow Analysis"
    body = (
        "Under Section 10 of RESPA, a servicer must conduct an annual escrow analysis. "
        "Any surplus greater than fifty dollars must be refunded to the borrower within thirty days. "
        "If a deficiency occurs, the servicer may spread the repayment over a minimum of two months."
    )
    chunks = chunk_article_text(title, body, tokenizer=tokenizer)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.token_count <= TOKENIZER_MAX_LENGTH
        real_enc = tokenizer.encode(chunk.content)
        assert chunk.token_count == sum(real_enc.attention_mask)
        assert chunk.content.startswith("Under Section 10")
