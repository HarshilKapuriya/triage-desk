"""Immutable Artifact Manifest and Integrity Verification for Ticket Embeddings.

Option A Scope:
This module defines the frozen upstream artifact specification, expected cryptographic
checksums, and integrity validation functions for sentence-transformers/all-MiniLM-L6-v2.

Operational and Cryptographic Notes:
1. Model Revision: Pinned to immutable upstream Git commit 1110a243fdf4706b3f48f1d95db1a4f5529b4d41.
2. Checksum Provenance:
   - onnx/model.onnx: Stored via Git LFS upstream. The SHA-256 digest matches the LFS OID.
   - Small config/vocab files: Stored as standard Git blobs upstream (not LFS). Their SHA-256
     digests were pre-computed directly from the raw byte contents of the pinned commit.
   - Pre-computed checksums guarantee exact byte-for-byte reproducibility against the pinned
     upstream commit. They do not constitute an independent third-party cryptographic audit
     or signature of the model weights.
3. Sequence Configuration:
   - sentence_bert_config.json configures the maximum sequence length (256 tokens), NOT the
     module pipeline sequence (which resides in modules.json).
4. Resource-Bounding Policy:
   - MAX_INPUT_CHARS (4,000 characters) is an application-level resource-bounding policy
     to protect CPU and memory from oversized payloads before tokenization.
   - Subword truncation is handled separately by the tokenizer at 256 tokens.
5. Direct vs Reference Artifacts:
   - Directly loaded at runtime: 'onnx/model.onnx' (InferenceSession) and 'tokenizer.json' (Tokenizer.from_file).
   - Integrity/provenance references: 'config.json', '1_Pooling/config.json', 'sentence_bert_config.json',
     and 'vocab.txt' are verified as part of the upstream bundle during provisioning, but are not directly
     read or parsed by the adapter during inference.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re

# Pinned model identity and versioning
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_DIM = 384
MAX_INPUT_CHARS = 4000
TOKENIZER_MAX_LENGTH = 256

# Explicit pipeline version tracking revision, lossy cap, truncation, pooling, and normalization
EMBEDDING_PIPELINE_VERSION = (
    f"{EMBEDDING_MODEL_NAME}@{EMBEDDING_MODEL_REVISION}"
    ":lossy4000chars:trunc256:meanpool:l2norm:v1"
)

DEFAULT_MODEL_DIR_REL = Path("models") / "sentence-transformers" / "all-MiniLM-L6-v2"


class ManifestError(Exception):
    """Base exception for manifest and artifact verification errors."""


class ArtifactMissingError(ManifestError):
    """Raised when one or more required model artifacts are missing from disk."""


class ArtifactCorruptedError(ManifestError):
    """Raised when an artifact's file size or SHA-256 digest does not match the manifest."""


@dataclass(frozen=True)
class ArtifactFileSpec:
    """Specification for a required model artifact file."""
    rel_path: str
    sha256: str
    size_bytes: int
    is_lfs: bool
    description: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ManifestError(
                f"Invalid SHA-256 format for {self.rel_path}: {self.sha256}"
            )
        if self.size_bytes <= 0:
            raise ManifestError(
                f"Invalid file size for {self.rel_path}: {self.size_bytes} bytes"
            )


# Complete manifest of required artifact files for offline inference
EMBEDDING_ARTIFACTS: dict[str, ArtifactFileSpec] = {
    "onnx/model.onnx": ArtifactFileSpec(
        rel_path="onnx/model.onnx",
        sha256="6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
        size_bytes=90405214,
        is_lfs=True,
        description="ONNX model graph weights (inputs: input_ids, attention_mask, token_type_ids; output: last_hidden_state)",
    ),
    "tokenizer.json": ArtifactFileSpec(
        rel_path="tokenizer.json",
        sha256="be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        size_bytes=466247,
        is_lfs=False,
        description="Fast Hugging Face tokenizer definition",
    ),
    "config.json": ArtifactFileSpec(
        rel_path="config.json",
        sha256="953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
        size_bytes=612,
        is_lfs=False,
        description="BERT transformer architecture configuration",
    ),
    "1_Pooling/config.json": ArtifactFileSpec(
        rel_path="1_Pooling/config.json",
        sha256="4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
        size_bytes=190,
        is_lfs=False,
        description="Pooling configuration specifying mean tokens pooling",
    ),
    "sentence_bert_config.json": ArtifactFileSpec(
        rel_path="sentence_bert_config.json",
        sha256="fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
        size_bytes=53,
        is_lfs=False,
        description="Sentence-BERT configuration defining max sequence length (256)",
    ),
    "vocab.txt": ArtifactFileSpec(
        rel_path="vocab.txt",
        sha256="07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        size_bytes=231508,
        is_lfs=False,
        description="WordPiece vocabulary tokens",
    ),
}


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute the hexadecimal SHA-256 digest of a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def validate_artifact_directory(
    directory: Path,
    artifacts: dict[str, ArtifactFileSpec] = EMBEDDING_ARTIFACTS,
) -> dict[str, Path]:
    """Validate that all required artifacts exist in the directory with correct sizes and digests.

    Returns:
        Mapping of relative artifact path to resolved Path on disk.

    Raises:
        ArtifactMissingError: If the directory or any required artifact is missing.
        ArtifactCorruptedError: If any artifact has incorrect size or SHA-256 hash.
    """
    resolved_dir = directory.resolve()
    if not resolved_dir.is_dir():
        raise ArtifactMissingError(
            f"Embedding model directory does not exist: {resolved_dir}"
        )

    resolved_paths: dict[str, Path] = {}
    for rel_path, spec in artifacts.items():
        file_path = resolved_dir / rel_path
        if not file_path.is_file():
            raise ArtifactMissingError(
                f"Missing required embedding artifact '{rel_path}' at {file_path}"
            )

        actual_size = file_path.stat().st_size
        if actual_size != spec.size_bytes:
            raise ArtifactCorruptedError(
                f"Artifact size mismatch for '{rel_path}': expected {spec.size_bytes} bytes, "
                f"found {actual_size} bytes at {file_path}"
            )

        actual_sha256 = compute_file_sha256(file_path)
        if actual_sha256 != spec.sha256:
            raise ArtifactCorruptedError(
                f"Artifact SHA-256 mismatch for '{rel_path}': expected {spec.sha256}, "
                f"found {actual_sha256} at {file_path}"
            )

        resolved_paths[rel_path] = file_path

    return resolved_paths
