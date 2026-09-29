"""Fetch and verify the private Run 8 checkpoint before the API starts."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch


CHECKPOINT_PATH = Path("/app/checkpoints_run8/final.pt")
EXPECTED_STATE_DICT_SHA256 = (
    "27e3b6a9cc5a0d4c092f836614d35be2dc54835bb1aa50c2a3b4060db66ea3be"
)
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024


def state_dict_hash(checkpoint_path: Path) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode("utf-8"))
        digest.update(state_dict[name].cpu().numpy().tobytes())
    return digest.hexdigest()


def verify(checkpoint_path: Path) -> None:
    actual = state_dict_hash(checkpoint_path)
    if actual != EXPECTED_STATE_DICT_SHA256:
        raise ValueError(
            "checkpoint weights do not match the verified Run 8 state-dictionary hash"
        )


def download_checkpoint(destination: Path, url: str) -> None:
    if urlparse(url).scheme != "https":
        raise ValueError("CHECKPOINT_URL must be a private HTTPS download URL")

    request_headers = {"User-Agent": "NexPath-checkpoint-bootstrap/1.0"}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and "huggingface.co" in url:
        request_headers["Authorization"] = f"Bearer {hf_token.strip()}"

    request = Request(url, headers=request_headers)
    temp_path: Path | None = None
    try:
        with urlopen(request, timeout=180) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=".checkpoint-", delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_CHECKPOINT_BYTES:
                        raise ValueError("checkpoint download exceeds the allowed size")
                    temp_file.write(chunk)

        verify(temp_path)
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> int:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_PATH.exists():
        try:
            verify(CHECKPOINT_PATH)
        except Exception as error:
            print(f"Checkpoint verification failed: {error}", file=sys.stderr)
            return 1
        print("Verified Run 8 checkpoint is present.")
        return 0

    url = os.environ.get("CHECKPOINT_URL", "").strip()
    if not url:
        print(
            "Run 8 checkpoint is missing. Set CHECKPOINT_URL to its private HTTPS artifact URL.",
            file=sys.stderr,
        )
        return 1

    try:
        print("Downloading the private Run 8 checkpoint...")
        download_checkpoint(CHECKPOINT_PATH, url)
    except HTTPError as error:
        print(
            f"Checkpoint download failed with HTTP {error.code}; check CHECKPOINT_URL access.",
            file=sys.stderr,
        )
        return 1
    except URLError:
        print("Checkpoint download failed; check network access and CHECKPOINT_URL.", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Checkpoint setup failed: {error}", file=sys.stderr)
        return 1

    print("Downloaded checkpoint matches the verified Run 8 weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
