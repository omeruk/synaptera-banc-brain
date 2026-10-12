"""Dataset adapters: raw release files -> :class:`connectome.schema.Connectome`."""
import hashlib
from pathlib import Path


def file_sha256(path) -> str:
    """SHA-256 of a file (streamed); recorded in Connectome.meta so every result names its exact input."""
    h = hashlib.sha256()
    with open(Path(path), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
