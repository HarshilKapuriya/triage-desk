"""Standalone Embedding Adapter for sentence-transformers/all-MiniLM-L6-v2 ONNX Model.

Option A Scope:
This module provides the core embedding inference pipeline without any worker, queue,
or database repository coupling.

Architectural and Operational Guarantees:
1. Provider: strictly CPUExecutionProvider only.
2. Artifact Verification: Validates complete artifact manifest (SHA-256 and sizes) before loading.
3. Graph Compatibility Check: On load, inspects session inputs and outputs to verify required
   names ('input_ids', 'attention_mask', 'token_type_ids' -> 'last_hidden_state'), tensor dtypes,
   and dimensions before running any inference.
4. Input Normalization: Strips whitespace, normalizes CRLF/CR to LF, and combines subject and body.
5. Resource-Bounding: Enforces a 4,000-character cap as an application-level resource-bounding policy
   before tokenization. Tokenizer truncation is configured at 256 subword tokens.
6. Mathematical Pooling & Normalization:
   - Attention-mask-weighted mean pooling across the sequence dimension.
   - Rejects zero-denominator masks without division by zero.
   - Rejects zero-norm vectors without division by zero.
   - Enforces L2 unit vector normalization with tolerance check (|norm - 1.0| <= 1e-5).
   - Enforces exactly 384 finite float components.
   - Never silently repairs invalid vectors into pseudo-valid embeddings.
"""

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from app.ml.manifest import (
    ArtifactCorruptedError,
    ArtifactMissingError,
    DEFAULT_MODEL_DIR_REL,
    EMBEDDING_ARTIFACTS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_PIPELINE_VERSION,
    MAX_INPUT_CHARS,
    TOKENIZER_MAX_LENGTH,
    validate_artifact_directory,
)

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base exception for embedding adapter errors."""


class IncompatibleModelError(EmbeddingError):
    """Raised when the ONNX model graph structure does not match the expected adapter contract."""


class InvalidEmbeddingError(EmbeddingError):
    """Raised when pooling, normalization, or output validation fails."""


@dataclass(frozen=True)
class EmbeddingResult:
    """Structured output from embedding generation."""
    vector: list[float]
    pipeline_version: str
    model_name: str
    model_revision: str
    char_count: int
    token_count: int


def normalize_and_format_text(subject: str | None, body: str | None) -> str:
    """Format and normalize subject and body into a single input string.

    Normalization steps:
    1. Replace Windows CRLF ('\r\n') and classic Mac CR ('\r') with standard LF ('\n').
    2. Strip leading and trailing whitespace from both components.
    3. Join non-empty subject and body with a double newline ('\n\n').
    4. Enforce non-empty input; raises ValueError if both components are empty or whitespace.
    5. Enforce MAX_INPUT_CHARS (4,000 characters) as an explicit lossy resource-bounding policy.
    """
    clean_subj = (subject or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    clean_body = (body or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    if clean_subj and clean_body:
        combined = f"{clean_subj}\n\n{clean_body}"
    elif clean_subj:
        combined = clean_subj
    elif clean_body:
        combined = clean_body
    else:
        raise ValueError("Ticket input text must contain non-empty subject or body.")

    # Explicit application resource-bounding policy
    if len(combined) > MAX_INPUT_CHARS:
        combined = combined[:MAX_INPUT_CHARS]

    return combined


def mean_pool_and_normalize(
    last_hidden_state: np.ndarray,
    attention_mask: np.ndarray,
    tolerance: float = 1e-5,
) -> list[float]:
    """Perform attention-mask-weighted mean pooling and L2 normalization.

    Args:
        last_hidden_state: Float array of shape (batch, seq_len, hidden_dim).
        attention_mask: Integer array of shape (batch, seq_len).
        tolerance: Allowed deviation from unit norm (1.0).

    Returns:
        List of 384 normalized floats.

    Raises:
        InvalidEmbeddingError: If shapes, mask sums, vector norms, or components are invalid.
    """
    # Shape verification
    if last_hidden_state.ndim != 3:
        raise InvalidEmbeddingError(
            f"Expected 3D last_hidden_state (batch, seq, dim), got ndim={last_hidden_state.ndim} shape={last_hidden_state.shape}"
        )
    if attention_mask.ndim != 2:
        raise InvalidEmbeddingError(
            f"Expected 2D attention_mask (batch, seq), got ndim={attention_mask.ndim} shape={attention_mask.shape}"
        )
    if last_hidden_state.shape[0] != 1 or attention_mask.shape[0] != 1:
        raise InvalidEmbeddingError(
            f"Adapter currently supports batch_size=1, got {last_hidden_state.shape[0]}"
        )
    if last_hidden_state.shape[1] != attention_mask.shape[1]:
        raise InvalidEmbeddingError(
            f"Sequence length mismatch: hidden={last_hidden_state.shape[1]}, mask={attention_mask.shape[1]}"
        )
    if last_hidden_state.shape[2] != EMBEDDING_DIM:
        raise InvalidEmbeddingError(
            f"Expected hidden dimension {EMBEDDING_DIM}, got {last_hidden_state.shape[2]}"
        )

    # Check finite hidden state components
    if not np.all(np.isfinite(last_hidden_state)):
        raise InvalidEmbeddingError("last_hidden_state contains NaN or infinite values.")

    # Expand attention mask to shape (1, seq_len, hidden_dim)
    mask_expanded = np.broadcast_to(
        np.expand_dims(attention_mask, -1),
        last_hidden_state.shape,
    ).astype(np.float32)

    # Sum token embeddings weighted by attention mask
    sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)  # shape (1, hidden_dim)

    # Sum of attention mask per sequence
    sum_mask = np.sum(mask_expanded, axis=1)  # shape (1, hidden_dim)

    # Nonzero denominator check
    if np.any(sum_mask <= 0):
        raise InvalidEmbeddingError(
            "Attention mask sum is zero or negative (no active unmasked tokens)."
        )

    pooled = sum_embeddings / sum_mask  # shape (1, hidden_dim)

    # L2 Norm calculation
    raw_norm = float(np.linalg.norm(pooled, ord=2))
    if not math.isfinite(raw_norm) or raw_norm <= 0:
        raise InvalidEmbeddingError(
            f"Calculated vector L2 norm is zero or non-finite: {raw_norm}"
        )

    normalized = (pooled / raw_norm)[0]  # shape (hidden_dim,)

    # Convert to standard Python float list
    vector = [float(x) for x in normalized]

    # Validate output length
    if len(vector) != EMBEDDING_DIM:
        raise InvalidEmbeddingError(
            f"Final vector length mismatch: expected {EMBEDDING_DIM}, got {len(vector)}"
        )

    # Validate finiteness of all components
    if not all(math.isfinite(x) for x in vector):
        raise InvalidEmbeddingError("Normalized vector contains NaN or infinite components.")

    # Validate unit length within tolerance
    actual_norm = math.sqrt(sum(x * x for x in vector))
    if abs(actual_norm - 1.0) > tolerance:
        raise InvalidEmbeddingError(
            f"Normalized vector does not have unit length: norm={actual_norm}, tolerance={tolerance}"
        )

    return vector


class EmbeddingModelAdapter:
    """Thread-safe, read-only embedding adapter wrapping Tokenizer and ONNX Runtime."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        tokenizer: Any | None = None,
        session: Any | None = None,
    ) -> None:
        """Initialize adapter.

        If tokenizer or session is provided, they are used directly (dependency injection for tests).
        Otherwise, validates the on-disk manifest and initializes ONNX session and Tokenizer.
        """
        self.model_name = EMBEDDING_MODEL_NAME
        self.model_revision = EMBEDDING_MODEL_REVISION
        self.pipeline_version = EMBEDDING_PIPELINE_VERSION
        self.embedding_dim = EMBEDDING_DIM

        if tokenizer is not None and session is not None:
            self._tokenizer = tokenizer
            self._session = session
            self._validate_session_contract(session)
            return

        resolved_dir = self._resolve_model_dir(model_dir)
        artifact_paths = validate_artifact_directory(resolved_dir)

        # 1. Initialize tokenizer
        tokenizer_path = artifact_paths["tokenizer.json"]
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=TOKENIZER_MAX_LENGTH)

        # 2. Initialize ONNX runtime session with CPUExecutionProvider strictly
        onnx_path = artifact_paths["onnx/model.onnx"]
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # 3. Validate graph inputs and outputs against contract
        self._validate_session_contract(self._session)

    @staticmethod
    def _resolve_model_dir(override_dir: str | Path | None) -> Path:
        if override_dir:
            return Path(override_dir).resolve()

        env_dir = os.getenv("EMBEDDING_MODEL_DIR")
        if env_dir:
            return Path(env_dir).resolve()

        # Try relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        candidate = project_root / DEFAULT_MODEL_DIR_REL
        if candidate.exists():
            return candidate.resolve()

        return Path(DEFAULT_MODEL_DIR_REL).resolve()

    def _validate_session_contract(self, session: Any) -> None:
        """Inspect ONNX session input and output metadata to ensure contract compatibility."""
        inputs = session.get_inputs()
        input_names = {inp.name for inp in inputs}
        expected_inputs = {"input_ids", "attention_mask", "token_type_ids"}

        missing_inputs = expected_inputs - input_names
        if missing_inputs:
            raise IncompatibleModelError(
                f"ONNX model missing required inputs: {sorted(missing_inputs)}. Found: {sorted(input_names)}"
            )

        for inp in inputs:
            if inp.name in expected_inputs:
                # Validate expected int64 datatype
                inp_type = getattr(inp, "type", "")
                if "int64" not in inp_type.lower():
                    raise IncompatibleModelError(
                        f"Expected int64 input tensor for '{inp.name}', got '{inp_type}'"
                    )

        outputs = session.get_outputs()
        output_names = {out.name for out in outputs}
        if "last_hidden_state" not in output_names:
            raise IncompatibleModelError(
                f"ONNX model missing required output 'last_hidden_state'. Found: {sorted(output_names)}"
            )

        for out in outputs:
            if out.name == "last_hidden_state":
                out_type = getattr(out, "type", "")
                if "float" not in out_type.lower():
                    raise IncompatibleModelError(
                        f"Expected float output tensor for 'last_hidden_state', got '{out_type}'"
                    )
                # Check shape dimension if available
                shape = getattr(out, "shape", None)
                if shape is not None:
                    if len(shape) != 3:
                        raise IncompatibleModelError(
                            f"Expected 3D output tensor [batch, seq, dim] for 'last_hidden_state', got rank {len(shape)}"
                        )
                    dim = shape[2]
                    if isinstance(dim, int) and dim != EMBEDDING_DIM:
                        raise IncompatibleModelError(
                            f"Expected hidden state dimension {EMBEDDING_DIM}, got {dim}"
                        )

    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate a 384-dimensional unit-normalized embedding for a plain-text passage.

        Args:
            text: Non-empty plain text passage (e.g. knowledge-base chunk).

        Returns:
            EmbeddingResult containing the unit-normalized vector and pipeline metadata.
        """
        clean_text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not clean_text:
            raise ValueError("Input text must not be empty.")
        if len(clean_text) > MAX_INPUT_CHARS:
            clean_text = clean_text[:MAX_INPUT_CHARS]

        encoding = self._tokenizer.encode(clean_text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        token_type_ids = np.array([encoding.type_ids], dtype=np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        outputs = self._session.run(["last_hidden_state"], ort_inputs)
        last_hidden_state = outputs[0]

        vector = mean_pool_and_normalize(
            last_hidden_state=last_hidden_state,
            attention_mask=attention_mask,
        )

        return EmbeddingResult(
            vector=vector,
            pipeline_version=self.pipeline_version,
            model_name=self.model_name,
            model_revision=self.model_revision,
            char_count=len(clean_text),
            token_count=len(encoding.ids),
        )

    def embed_ticket(self, subject: str | None, body: str | None) -> EmbeddingResult:
        """Generate a 384-dimensional unit-normalized embedding for a ticket.

        Args:
            subject: Ticket subject line or title.
            body: Ticket body or complaint narrative.

        Returns:
            EmbeddingResult containing the unit-normalized vector and pipeline metadata.
        """
        formatted_text = normalize_and_format_text(subject=subject, body=body)
        return self.embed_text(formatted_text)
