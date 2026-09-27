"""Deterministic metadata tag encoding for KacheDB 64-bit bitmask pre-filtering.

Fallback module for environments where kachedb-py < 0.2.0 is installed.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def tag_to_bit(tag: str) -> int:
    """Deterministically map a tag string to a bit offset 0..63 using SHA-256."""
    clean_tag = tag.strip().lower()
    digest = hashlib.sha256(clean_tag.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % 64


def tags_to_bitmask(
    tags: Iterable[str] | None,
    base_mask: int = 0,
) -> int:
    """Encode an iterable of tag strings into a single 64-bit integer bitmask."""
    mask = base_mask & 0xFFFFFFFFFFFFFFFF
    if tags is not None:
        for t in tags:
            if t and t.strip():
                bit = tag_to_bit(t)
                mask |= 1 << bit
    return mask & 0xFFFFFFFFFFFFFFFF
