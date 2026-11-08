"""Comprehensive Offline Unit Tests for Embedding Manifest, Downloader, and Adapter.

Option A Scope:
Exercises adapter logic, text preprocessing, pooling mathematics, vector normalization,
graph compatibility checks, manifest validation, and atomic downloader behavior using
purely offline mocks and fakes. No network requests or real model downloads are performed.
"""

from dataclasses import dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from app.ml.embeddings import (
    EmbeddingModelAdapter,
    EmbeddingResult,
    IncompatibleModelError,
    InvalidEmbeddingError,
    mean_pool_and_normalize,
    normalize_and_format_text,
)
from app.ml.manifest import (
    ArtifactCorruptedError,
    ArtifactFileSpec,
    ArtifactMissingError,
    DEFAULT_MODEL_DIR_REL,
    EMBEDDING_ARTIFACTS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_PIPELINE_VERSION,
    MAX_INPUT_CHARS,
    ManifestError,
    TOKENIZER_MAX_LENGTH,
    compute_file_sha256,
    validate_artifact_directory,
)
from scripts.download_embedding_model import (
    DownloadError,
    download_single_artifact,
    provision_model_artifacts,
)


# ============================================================================
# Fakes and Mocks for Offline Boundary Testing
# ============================================================================

@dataclass
class FakeEncoding:
    """Fake encoding returned by MockTokenizer."""
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


class FakeTokenizer:
    """Deterministic offline fake tokenizer."""

    def __init__(self, token_count: int = 10) -> None:
        self.truncation_max_length: int | None = None
        self.token_count = token_count

    def enable_truncation(self, max_length: int) -> None:
        self.truncation_max_length = max_length

    def encode(self, text: str) -> FakeEncoding:
        count = min(self.token_count, self.truncation_max_length or self.token_count)
        return FakeEncoding(
            ids=list(range(100, 100 + count)),
            attention_mask=[1] * count,
            type_ids=[0] * count,
        )


class FakeNodeMeta:
    """Mock for ONNX input/output node metadata."""

    def __init__(self, name: str, node_type: str = "tensor(int64)", shape: list[Any] | None = None) -> None:
        self.name = name
        self.type = node_type
        self.shape = shape or [1, None, EMBEDDING_DIM]


class FakeInferenceSession:
    """Deterministic offline fake ONNX runtime session."""

    def __init__(
        self,
        inputs: list[FakeNodeMeta] | None = None,
        outputs: list[FakeNodeMeta] | None = None,
        return_tensor: np.ndarray | None = None,
    ) -> None:
        self._inputs = inputs or [
            FakeNodeMeta("input_ids", "tensor(int64)"),
            FakeNodeMeta("attention_mask", "tensor(int64)"),
            FakeNodeMeta("token_type_ids", "tensor(int64)"),
        ]
        self._outputs = outputs or [
            FakeNodeMeta("last_hidden_state", "tensor(float)", [1, None, EMBEDDING_DIM]),
        ]
        self.return_tensor = return_tensor
        self.last_run_inputs: dict[str, Any] | None = None

    def get_inputs(self) -> list[FakeNodeMeta]:
        return self._inputs

    def get_outputs(self) -> list[FakeNodeMeta]:
        return self._outputs

    def run(self, output_names: list[str], input_feed: dict[str, Any]) -> list[np.ndarray]:
        self.last_run_inputs = input_feed
        if self.return_tensor is not None:
            return [self.return_tensor]

        # Generate a valid default 3D tensor: shape (1, seq_len, 384)
        seq_len = input_feed["input_ids"].shape[1]
        rng = np.random.RandomState(42)
        tensor = rng.randn(1, seq_len, EMBEDDING_DIM).astype(np.float32)
        return [tensor]


# ============================================================================
# Section 1: Text Normalization and Formatting Tests
# ============================================================================

def test_normalize_and_format_text_clean_joining():
    text = normalize_and_format_text("Subject Title", "Body narrative content.")
    assert text == "Subject Title\n\nBody narrative content."


def test_normalize_and_format_text_newline_normalization():
    # CRLF (\r\n) and classic CR (\r) are normalized to LF (\n)
    text = normalize_and_format_text("Subject\r\nLine 1\rLine 2", "Body\r\nParagraph 1\rParagraph 2")
    assert "\r" not in text
    assert text == "Subject\nLine 1\nLine 2\n\nBody\nParagraph 1\nParagraph 2"


def test_normalize_and_format_text_whitespace_stripping():
    text = normalize_and_format_text("   Subject with spaces   ", "   Body with spaces   \n")
    assert text == "Subject with spaces\n\nBody with spaces"


def test_normalize_and_format_text_single_field_handling():
    assert normalize_and_format_text("Only Subject", None) == "Only Subject"
    assert normalize_and_format_text("Only Subject", "") == "Only Subject"
    assert normalize_and_format_text(None, "Only Body") == "Only Body"
    assert normalize_and_format_text("", "Only Body") == "Only Body"


def test_normalize_and_format_text_empty_input_rejected():
    with pytest.raises(ValueError, match="non-empty subject or body"):
        normalize_and_format_text("", "")

    with pytest.raises(ValueError, match="non-empty subject or body"):
        normalize_and_format_text("   \r\n  ", None)

    with pytest.raises(ValueError, match="non-empty subject or body"):
        normalize_and_format_text(None, None)


def test_normalize_and_format_text_resource_bounding_cap():
    # Enforces 4,000-character cap as an application resource-bounding policy
    long_subject = "A" * 2000
    long_body = "B" * 3000
    combined = normalize_and_format_text(long_subject, long_body)
    assert len(combined) == MAX_INPUT_CHARS
    assert combined.startswith("A" * 2000 + "\n\n" + "B" * 1998)


# ============================================================================
# Section 2: Mean Pooling and L2 Normalization Mathematics Tests
# ============================================================================

def test_mean_pool_and_normalize_mathematical_precision():
    # Construct exact 2-token sequence where token 1 is active (mask=1) and token 2 is padded (mask=0)
    hidden_dim = EMBEDDING_DIM
    last_hidden = np.zeros((1, 2, hidden_dim), dtype=np.float32)
    # Token 0: vector of 3.0s
    last_hidden[0, 0, :] = 3.0
    # Token 1: vector of 999.0s (padding token that MUST be ignored)
    last_hidden[0, 1, :] = 999.0

    attention_mask = np.array([[1, 0]], dtype=np.int64)

    vector = mean_pool_and_normalize(last_hidden, attention_mask)
    assert len(vector) == hidden_dim

    # Since token 0 is constant 3.0 across all 384 dims, the mean pool is constant 3.0.
    # Unit normalization of a constant vector of dimension D gives 1 / sqrt(D) for all components.
    expected_component = 1.0 / math.sqrt(hidden_dim)
    for comp in vector:
        assert math.isclose(comp, expected_component, rel_tol=1e-5)

    # Unit norm verification
    norm = math.sqrt(sum(x * x for x in vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


def test_mean_pool_and_normalize_multi_token_weighted_average():
    hidden_dim = EMBEDDING_DIM
    last_hidden = np.zeros((1, 3, hidden_dim), dtype=np.float32)
    # Token 0: 2.0, Token 1: 4.0 -> average should be 3.0. Token 2 is masked (mask=0).
    last_hidden[0, 0, :] = 2.0
    last_hidden[0, 1, :] = 4.0
    last_hidden[0, 2, :] = 500.0

    attention_mask = np.array([[1, 1, 0]], dtype=np.int64)

    vector = mean_pool_and_normalize(last_hidden, attention_mask)
    expected_component = 1.0 / math.sqrt(hidden_dim)
    for comp in vector:
        assert math.isclose(comp, expected_component, rel_tol=1e-5)


def test_mean_pool_and_normalize_rejects_empty_mask():
    hidden_dim = EMBEDDING_DIM
    last_hidden = np.ones((1, 2, hidden_dim), dtype=np.float32)
    # Both tokens masked out (sum of mask = 0)
    attention_mask = np.array([[0, 0]], dtype=np.int64)

    with pytest.raises(InvalidEmbeddingError, match="Attention mask sum is zero"):
        mean_pool_and_normalize(last_hidden, attention_mask)


def test_mean_pool_and_normalize_rejects_zero_norm():
    hidden_dim = EMBEDDING_DIM
    # All zeros produces zero norm
    last_hidden = np.zeros((1, 2, hidden_dim), dtype=np.float32)
    attention_mask = np.array([[1, 1]], dtype=np.int64)

    with pytest.raises(InvalidEmbeddingError, match="vector L2 norm is zero"):
        mean_pool_and_normalize(last_hidden, attention_mask)


def test_mean_pool_and_normalize_rejects_nan_and_inf():
    hidden_dim = EMBEDDING_DIM
    last_hidden = np.ones((1, 2, hidden_dim), dtype=np.float32)
    last_hidden[0, 0, 10] = np.nan
    attention_mask = np.array([[1, 1]], dtype=np.int64)

    with pytest.raises(InvalidEmbeddingError, match="contains NaN or infinite"):
        mean_pool_and_normalize(last_hidden, attention_mask)

    last_hidden[0, 0, 10] = np.inf
    with pytest.raises(InvalidEmbeddingError, match="contains NaN or infinite"):
        mean_pool_and_normalize(last_hidden, attention_mask)


def test_mean_pool_and_normalize_rejects_wrong_dimensions():
    # 2D tensor instead of 3D
    with pytest.raises(InvalidEmbeddingError, match="Expected 3D last_hidden_state"):
        mean_pool_and_normalize(np.ones((1, 384), dtype=np.float32), np.ones((1, 10), dtype=np.int64))

    # Wrong hidden dim (512 instead of 384)
    with pytest.raises(InvalidEmbeddingError, match=f"Expected hidden dimension {EMBEDDING_DIM}"):
        mean_pool_and_normalize(np.ones((1, 5, 512), dtype=np.float32), np.ones((1, 5), dtype=np.int64))


# ============================================================================
# Section 3: Adapter Contract, Injection, and Graph Inspection Tests
# ============================================================================

def test_adapter_embed_ticket_success_with_fakes():
    tokenizer = FakeTokenizer(token_count=8)
    session = FakeInferenceSession()

    adapter = EmbeddingModelAdapter(tokenizer=tokenizer, session=session)
    result = adapter.embed_ticket("Credit Card Dispute", "Fraudulent transaction reported.")

    assert isinstance(result, EmbeddingResult)
    assert len(result.vector) == EMBEDDING_DIM
    assert result.pipeline_version == EMBEDDING_PIPELINE_VERSION
    assert result.model_name == EMBEDDING_MODEL_NAME
    assert result.model_revision == EMBEDDING_MODEL_REVISION
    assert result.token_count == 8

    # Verify input fed to ONNX session
    last_inputs = session.last_run_inputs
    assert last_inputs is not None
    assert "input_ids" in last_inputs
    assert "attention_mask" in last_inputs
    assert "token_type_ids" in last_inputs
    assert last_inputs["input_ids"].dtype == np.int64
    assert last_inputs["input_ids"].shape == (1, 8)


def test_adapter_rejects_missing_onnx_input_names():
    tokenizer = FakeTokenizer()
    # Missing 'token_type_ids'
    session = FakeInferenceSession(
        inputs=[
            FakeNodeMeta("input_ids", "tensor(int64)"),
            FakeNodeMeta("attention_mask", "tensor(int64)"),
        ]
    )
    with pytest.raises(IncompatibleModelError, match="missing required inputs"):
        EmbeddingModelAdapter(tokenizer=tokenizer, session=session)


def test_adapter_rejects_wrong_onnx_input_dtype():
    tokenizer = FakeTokenizer()
    # input_ids has type tensor(float) instead of tensor(int64)
    session = FakeInferenceSession(
        inputs=[
            FakeNodeMeta("input_ids", "tensor(float)"),
            FakeNodeMeta("attention_mask", "tensor(int64)"),
            FakeNodeMeta("token_type_ids", "tensor(int64)"),
        ]
    )
    with pytest.raises(IncompatibleModelError, match="Expected int64 input tensor"):
        EmbeddingModelAdapter(tokenizer=tokenizer, session=session)


def test_adapter_rejects_missing_onnx_output_name():
    tokenizer = FakeTokenizer()
    # Output named 'pooled_output' instead of 'last_hidden_state'
    session = FakeInferenceSession(
        outputs=[FakeNodeMeta("pooled_output", "tensor(float)", [1, 384])]
    )
    with pytest.raises(IncompatibleModelError, match="missing required output 'last_hidden_state'"):
        EmbeddingModelAdapter(tokenizer=tokenizer, session=session)


def test_adapter_rejects_wrong_onnx_output_dim():
    tokenizer = FakeTokenizer()
    # Output dim is 768 instead of 384
    session = FakeInferenceSession(
        outputs=[FakeNodeMeta("last_hidden_state", "tensor(float)", [1, None, 768])]
    )
    with pytest.raises(IncompatibleModelError, match="Expected hidden state dimension 384, got 768"):
        EmbeddingModelAdapter(tokenizer=tokenizer, session=session)


def test_adapter_rejects_non_3d_onnx_output_rank():
    tokenizer = FakeTokenizer()
    # 2D output [1, 384] instead of 3D [1, seq, 384]
    session_2d = FakeInferenceSession(
        outputs=[FakeNodeMeta("last_hidden_state", "tensor(float)", [1, 384])]
    )
    with pytest.raises(IncompatibleModelError, match="Expected 3D output tensor.*got rank 2"):
        EmbeddingModelAdapter(tokenizer=tokenizer, session=session_2d)

    # 4D output [1, 10, 384, 1] instead of 3D
    session_4d = FakeInferenceSession(
        outputs=[FakeNodeMeta("last_hidden_state", "tensor(float)", [1, 10, 384, 1])]
    )
    with pytest.raises(IncompatibleModelError, match="Expected 3D output tensor.*got rank 4"):
        EmbeddingModelAdapter(tokenizer=tokenizer, session=session_4d)


def test_manifest_pipeline_identity_contains_full_revision():
    # Pipeline identity must contain full 40-character revision, lossy cap, truncation, and pooling
    assert len(EMBEDDING_MODEL_REVISION) == 40
    assert EMBEDDING_MODEL_REVISION in EMBEDDING_PIPELINE_VERSION
    assert EMBEDDING_PIPELINE_VERSION == (
        f"{EMBEDDING_MODEL_NAME}@{EMBEDDING_MODEL_REVISION}"
        ":lossy4000chars:trunc256:meanpool:l2norm:v1"
    )


# ============================================================================
# Section 4: Manifest and Artifact Validation Tests
# ============================================================================

def test_manifest_specs_are_valid():
    for rel_path, spec in EMBEDDING_ARTIFACTS.items():
        assert len(spec.sha256) == 64
        assert int(spec.sha256, 16) > 0
        assert spec.size_bytes > 0


def test_manifest_spec_rejects_invalid_sha256():
    with pytest.raises(ManifestError, match="Invalid SHA-256 format"):
        ArtifactFileSpec(
            rel_path="test.bin",
            sha256="not_a_valid_hex_string",
            size_bytes=100,
            is_lfs=False,
            description="test",
        )


def test_validate_artifact_directory_missing_dir(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(ArtifactMissingError, match="does not exist"):
        validate_artifact_directory(non_existent)


def test_validate_artifact_directory_missing_files(tmp_path: Path):
    empty_dir = tmp_path / "empty_model_dir"
    empty_dir.mkdir()
    with pytest.raises(ArtifactMissingError, match="Missing required embedding artifact"):
        validate_artifact_directory(empty_dir)


def test_validate_artifact_directory_size_mismatch(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Create dummy spec
    dummy_spec = ArtifactFileSpec("sample.txt", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 100, False, "test")
    test_file = model_dir / "sample.txt"
    test_file.write_bytes(b"hello")  # only 5 bytes, expected 100

    with pytest.raises(ArtifactCorruptedError, match="size mismatch"):
        validate_artifact_directory(model_dir, {"sample.txt": dummy_spec})


def test_validate_artifact_directory_checksum_mismatch(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # 5 bytes written, expected 5 bytes, but hash is for empty file
    dummy_spec = ArtifactFileSpec("sample.txt", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 5, False, "test")
    test_file = model_dir / "sample.txt"
    test_file.write_bytes(b"hello")

    with pytest.raises(ArtifactCorruptedError, match="SHA-256 mismatch"):
        validate_artifact_directory(model_dir, {"sample.txt": dummy_spec})


# ============================================================================
# Section 5: Provisioning Downloader Unit Tests (Mocked Network)
# ============================================================================

def test_downloader_atomic_success_with_mocked_network(tmp_path: Path):
    target_dir = tmp_path / "models"
    content = b"Mocked artifact content for testing download streaming."
    content_sha256 = hashlib.sha256(content).hexdigest()
    spec = ArtifactFileSpec("test_file.bin", content_sha256, len(content), False, "test")

    # Mock httpx response streaming
    def mock_stream(method, url, follow_redirects=True, timeout=60.0):
        class MockStreamContext:
            def __enter__(self):
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.iter_bytes = MagicMock(return_value=[content[:10], content[10:]])
                return mock_resp
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return MockStreamContext()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.stream = mock_stream

    result_path = download_single_artifact(
        client=mock_client,
        spec=spec,
        target_dir=target_dir,
    )

    assert result_path.is_file()
    assert result_path.read_bytes() == content
    assert compute_file_sha256(result_path) == content_sha256


def test_downloader_checksum_mismatch_cleans_up_and_does_not_replace(tmp_path: Path):
    target_dir = tmp_path / "models"
    target_file = target_dir / "test_file.bin"
    target_file.parent.mkdir(parents=True)
    original_valid_content = b"ORIGINAL_VALID_CONTENT"
    target_file.write_bytes(original_valid_content)

    # Download returns corrupted content of the same size
    corrupt_content = b"X" * len(original_valid_content)
    expected_sha256 = hashlib.sha256(original_valid_content).hexdigest()
    spec = ArtifactFileSpec("test_file.bin", expected_sha256, len(original_valid_content), False, "test")

    def mock_stream(method, url, follow_redirects=True, timeout=60.0):
        class MockStreamContext:
            def __enter__(self):
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.iter_bytes = MagicMock(return_value=[corrupt_content])
                return mock_resp
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return MockStreamContext()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.stream = mock_stream

    with pytest.raises(DownloadError, match="Checksum mismatch"):
        download_single_artifact(
            client=mock_client,
            spec=spec,
            target_dir=target_dir,
            force=True,
        )

    # Pre-existing file MUST NOT be overwritten
    assert target_file.read_bytes() == original_valid_content
    # Temporary files MUST be cleaned up
    temp_files = list(target_dir.glob(".tmp_*"))
    assert len(temp_files) == 0


def test_downloader_oversized_response_aborts_and_cleans_up(tmp_path: Path):
    target_dir = tmp_path / "models"
    spec = ArtifactFileSpec("test_file.bin", "a" * 64, 10, False, "test")

    def mock_stream(method, url, follow_redirects=True, timeout=60.0):
        class MockStreamContext:
            def __enter__(self):
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                # Streams 20 bytes when max size is 10
                mock_resp.iter_bytes = MagicMock(return_value=[b"A" * 15, b"B" * 5])
                return mock_resp
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return MockStreamContext()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.stream = mock_stream

    with pytest.raises(DownloadError, match="Oversized download"):
        download_single_artifact(
            client=mock_client,
            spec=spec,
            target_dir=target_dir,
        )

    temp_files = list(target_dir.glob(".tmp_*"))
    assert len(temp_files) == 0


def test_downloader_truncated_response_aborts_and_cleans_up(tmp_path: Path):
    target_dir = tmp_path / "models"
    spec = ArtifactFileSpec("test_file.bin", "a" * 64, 50, False, "test")

    def mock_stream(method, url, follow_redirects=True, timeout=60.0):
        class MockStreamContext:
            def __enter__(self):
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                # Streams only 10 bytes when expected size is 50
                mock_resp.iter_bytes = MagicMock(return_value=[b"A" * 10])
                return mock_resp
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return MockStreamContext()

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.stream = mock_stream

    with pytest.raises(DownloadError, match="Truncated download"):
        download_single_artifact(
            client=mock_client,
            spec=spec,
            target_dir=target_dir,
        )

    temp_files = list(target_dir.glob(".tmp_*"))
    assert len(temp_files) == 0


# ============================================================================
# Section 6: Opt-in Real Model Smoke Test
# ============================================================================

@pytest.mark.skipif(
    os.getenv("RUN_REAL_EMBEDDING_TESTS", "").lower() != "true",
    reason="Opt-in real embedding test requires RUN_REAL_EMBEDDING_TESTS=true",
)
def test_real_embedding_model_smoke():
    """Opt-in real model smoke test.

    Contract:
    - Skipped by default.
    - If RUN_REAL_EMBEDDING_TESTS=true is set:
      - Validates that real model artifacts exist and match manifest digests.
      - If missing or corrupt, fails with a clear failure message directing operator to run provisioning.
      - If present, executes real inference and validates 384 dimensions and unit norm.
    """
    model_dir = Path(__file__).resolve().parent.parent / DEFAULT_MODEL_DIR_REL
    try:
        adapter = EmbeddingModelAdapter(model_dir=model_dir)
    except (ArtifactMissingError, ArtifactCorruptedError) as err:
        pytest.fail(
            f"RUN_REAL_EMBEDDING_TESTS=true but embedding model artifacts are missing or corrupt: {err}\n"
            f"Run 'python scripts/download_embedding_model.py' to provision artifacts."
        )

    result = adapter.embed_ticket(
        subject="Unauthorized transfer from checking account",
        body="I noticed a $500 withdrawal on my checking account that I did not authorize.",
    )
    assert len(result.vector) == EMBEDDING_DIM
    norm = math.sqrt(sum(x * x for x in result.vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)
    assert all(math.isfinite(x) for x in result.vector)
