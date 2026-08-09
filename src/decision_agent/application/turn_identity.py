"""Deterministic, non-reversible identifiers for persisted session turns."""

from __future__ import annotations

from hashlib import sha256

_TURN_ID_PREFIX = "turn-v1-"
_ENCODING = "utf-8"


def derive_turn_id(session_id: str, request_id: str) -> str:
    """Derive a stable turn ID without exposing either input identifier.

    Length-prefixed UTF-8 fields make the hash input unambiguous even when request
    IDs contain arbitrary printable punctuation.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be non-empty")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be non-empty")
    session_bytes = session_id.encode(_ENCODING)
    request_bytes = request_id.encode(_ENCODING)
    payload = b"turn-v1\x00" + _encode_field(session_bytes) + _encode_field(request_bytes)
    return _TURN_ID_PREFIX + sha256(payload).hexdigest()


def _encode_field(value: bytes) -> bytes:
    return len(value).to_bytes(8, byteorder="big") + value
