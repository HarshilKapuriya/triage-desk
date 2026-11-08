"""Explicit Provisioning CLI for sentence-transformers/all-MiniLM-L6-v2 ONNX Model.

Operational Boundaries:
1. Explicit CLI only: This script must be invoked explicitly by an operator or CI provisioning step.
   The model adapter never downloads files at runtime or import time.
2. Safety and Atomicity:
   - Streams downloads to temporary files (.tmp_*) on disk.
   - Enforces connection and read timeouts.
   - Computes SHA-256 and validates file sizes during streaming.
   - Immediately aborts and cleans up temp files if downloaded size exceeds manifest expected size.
   - Cleans up temporary files on any failure, connection drop, or mismatch.
   - Never overwrites an existing valid artifact with an incomplete or invalid file.
   - Atomicity scope: Atomicity is strictly per-file (via os.replace on temporary files),
     NOT whole-bundle. Each file is verified and atomically replaced individually; a failed
     download cleans up its own temp file without rolling back previously verified files.
"""

import argparse
import hashlib
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import httpx

# Ensure project root is in sys.path when running as script
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.ml.manifest import (
    ArtifactCorruptedError,
    ArtifactFileSpec,
    ArtifactMissingError,
    DEFAULT_MODEL_DIR_REL,
    EMBEDDING_ARTIFACTS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    validate_artifact_directory,
)

logger = logging.getLogger("download_embedding_model")

HF_RESOLVE_BASE_URL = f"https://huggingface.co/{EMBEDDING_MODEL_NAME}/resolve/{EMBEDDING_MODEL_REVISION}"


class DownloadError(Exception):
    """Raised when an artifact download or verification fails."""


def download_single_artifact(
    client: httpx.Client,
    spec: ArtifactFileSpec,
    target_dir: Path,
    force: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Download and verify a single artifact file, replacing target atomically on success.

    Guarantees:
    - Never overwrites a valid file if download fails or produces invalid hash.
    - Aborts if byte count exceeds expected size.
    - Cleans up temporary files on failure.
    """
    dest_path = target_dir / spec.rel_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if target already exists and is fully valid
    if dest_path.is_file() and not force:
        if dest_path.stat().st_size == spec.size_bytes:
            hasher = hashlib.sha256()
            with open(dest_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            if hasher.hexdigest().lower() == spec.sha256:
                logger.info("Artifact already valid, skipping: %s", spec.rel_path)
                return dest_path

    url = f"{HF_RESOLVE_BASE_URL}/{spec.rel_path}"
    logger.info("Downloading %s from %s", spec.rel_path, url)

    # Create temporary file in the target directory for atomic same-filesystem os.replace
    temp_file = tempfile.NamedTemporaryFile(
        dir=dest_path.parent,
        prefix=".tmp_download_",
        delete=False,
    )
    temp_path = Path(temp_file.name)

    bytes_written = 0
    hasher = hashlib.sha256()

    try:
        with temp_file:
            with client.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
                if response.status_code != 200:
                    raise DownloadError(
                        f"HTTP {response.status_code} while downloading {spec.rel_path} from {url}"
                    )

                for chunk in response.iter_bytes(chunk_size=65536):
                    bytes_written += len(chunk)
                    if bytes_written > spec.size_bytes:
                        raise DownloadError(
                            f"Oversized download for {spec.rel_path}: received {bytes_written} bytes, "
                            f"exceeding expected limit of {spec.size_bytes} bytes"
                        )
                    hasher.update(chunk)
                    temp_file.write(chunk)

        # Verify final size
        if bytes_written != spec.size_bytes:
            raise DownloadError(
                f"Truncated download for {spec.rel_path}: received {bytes_written} bytes, "
                f"expected {spec.size_bytes} bytes"
            )

        # Verify final checksum
        actual_sha256 = hasher.hexdigest().lower()
        if actual_sha256 != spec.sha256:
            raise DownloadError(
                f"Checksum mismatch for {spec.rel_path}: expected {spec.sha256}, "
                f"got {actual_sha256}"
            )

        # Atomic replacement: only executed when size and digest are 100% verified
        os.replace(temp_path, dest_path)
        logger.info("Successfully provisioned %s (%d bytes)", spec.rel_path, bytes_written)
        return dest_path

    except Exception:
        # Guarantee cleanup of temporary file on any error
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def provision_model_artifacts(
    target_dir: Path,
    force: bool = False,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> dict[str, Path]:
    """Download all artifacts specified in EMBEDDING_ARTIFACTS and validate the full directory."""
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    owns_client = client is None
    http_client = client or httpx.Client()

    try:
        for rel_path, spec in EMBEDDING_ARTIFACTS.items():
            download_single_artifact(
                client=http_client,
                spec=spec,
                target_dir=target_dir,
                force=force,
                timeout=timeout,
            )

        # Full verification of complete directory
        return validate_artifact_directory(target_dir)
    finally:
        if owns_client:
            http_client.close()


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Download and verify sentence-transformers/all-MiniLM-L6-v2 ONNX model artifacts."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=project_root / DEFAULT_MODEL_DIR_REL,
        help="Target directory to store model artifacts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite even if local artifacts are valid.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Network timeout in seconds per request.",
    )

    args = parser.parse_args()

    try:
        provision_model_artifacts(target_dir=args.target_dir, force=args.force, timeout=args.timeout)
        logger.info("All embedding model artifacts successfully verified at: %s", args.target_dir)
        return 0
    except (DownloadError, ManifestError) as err:
        logger.error("Provisioning failed: %s", err)
        return 1
    except Exception as err:
        logger.exception("Unexpected error during provisioning: %s", err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
