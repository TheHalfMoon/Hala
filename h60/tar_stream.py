"""H60 incremental, boundary-safe TAR member reader for one original source TAR.

The reader consumes an arbitrary byte stream in order and emits complete
regular-file members. Member payloads that straddle a chunk or a pinned
512 MiB source window are never split, never truncated and never mistaken for
a finished example. Header checksum validation rejects byte patterns that
merely look like a TAR header inside JPEG or JSON payload bytes.
"""
from __future__ import annotations

import re

BLOCK = 512
MAX_MEMBER_BYTES = 32 * 1024 * 1024
UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_USTAR_MAGIC = b"ustar"


def _octal(field: bytes) -> int | None:
    text = field.split(b"\0")[0].strip()
    if not text:
        return 0
    try:
        return int(text, 8)
    except ValueError:
        return None


def _size(field: bytes) -> int:
    # GNU base-256 encoding for large values.
    if field and field[0] & 0x80:
        value = field[0] & 0x7F
        for byte in field[1:]:
            value = (value << 8) | byte
        return value
    parsed = _octal(field)
    if parsed is None:
        raise ValueError("original TAR header has a malformed size field")
    return parsed


def _checksum_ok(block: bytes) -> bool:
    if len(block) != BLOCK:
        return False
    stored = _octal(block[148:156])
    if stored is None:
        return False
    unsigned = sum(block[:148]) + 256 + sum(block[156:])
    signed = (
        sum(byte - 256 if byte > 127 else byte for byte in block[:148])
        + 256
        + sum(byte - 256 if byte > 127 else byte for byte in block[156:])
    )
    return stored in (unsigned, signed)


def _member_name(block: bytes) -> str:
    name = block[:100].split(b"\0")[0]
    prefix = block[345:500].split(b"\0")[0]
    if prefix:
        name = prefix + b"/" + name
    return name.decode("utf8", "replace")


def _pax_path(payload: bytes) -> str | None:
    """Extract `path=` from a pax extended header payload."""
    offset = 0
    result = None
    while offset < len(payload):
        space = payload.find(b" ", offset)
        if space < 0:
            break
        try:
            length = int(payload[offset:space])
        except ValueError:
            break
        if length <= 0 or offset + length > len(payload):
            break
        record = payload[space + 1 : offset + length].rstrip(b"\n")
        if record.startswith(b"path="):
            result = record[5:].decode("utf8", "replace")
        offset += length
    return result


class TarMember:
    __slots__ = ("name", "size", "offset", "typeflag")

    def __init__(self, name: str, size: int, offset: int, typeflag: bytes):
        self.name = name
        self.size = size
        self.offset = offset
        self.typeflag = typeflag

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "size": self.size,
            "offset": self.offset,
            "typeflag": self.typeflag.decode("ascii", "replace"),
        }


class TarStreamReader:
    """Stream a POSIX/GNU TAR and report complete regular members.

    `predicate(member)` decides whether a member is retained; retained members
    are handed to `sink(member, payload)` exactly once, with the full payload.
    """

    def __init__(self, predicate, sink):
        self._predicate = predicate
        self._sink = sink
        self._buffer = bytearray()
        self._base = 0
        self._state = "header"
        self._member: TarMember | None = None
        self._remaining = 0
        self._payload = bytearray()
        self._retain = False
        self._long_name: str | None = None
        self._pax_name: str | None = None
        self._pad = 0
        self._finished = False
        self.members_seen = 0
        self.members_emitted = 0
        self.bytes_consumed = 0
        self.rejected_headers = 0

    # -- public ---------------------------------------------------------
    def feed(self, chunk: bytes) -> None:
        if self._finished:
            raise ValueError("TAR stream already finished")
        if chunk:
            self._buffer += chunk
            self.bytes_consumed += len(chunk)
        self._drain()

    def close(self) -> None:
        """Finish the stream; only an exact end-of-archive is accepted."""
        self._drain()
        if not self._finished:
            raise ValueError("TAR stream ended without a complete end-of-archive marker")
        if self._state == "data" and self._remaining:
            raise ValueError("TAR stream ended inside a member payload")

    @property
    def finished(self) -> bool:
        return self._finished

    # -- internals ------------------------------------------------------
    def _drain(self) -> None:
        while self._state != "done":
            if self._state == "header":
                if not self._consume_header():
                    return
            elif self._state == "data":
                if not self._consume_data():
                    return
            elif self._state == "padding":
                if not self._consume_padding():
                    return
            elif self._state == "longname" or self._state == "pax":
                if not self._consume_extended():
                    return
            elif self._state == "skip":
                if not self._consume_skip():
                    return
            else:  # pragma: no cover - defensive
                raise ValueError("unknown TAR reader state")

    def _consume_header(self) -> bool:
        if len(self._buffer) < BLOCK:
            return False
        block = bytes(self._buffer[:BLOCK])
        if block == b"\0" * BLOCK:
            if len(self._buffer) < 2 * BLOCK:
                return False
            del self._buffer[: 2 * BLOCK]
            self._base += 2 * BLOCK
            self._state = "done"
            self._finished = True
            return True
        if not _checksum_ok(block) or block[257:262] != _USTAR_MAGIC:
            # Not a header: payload noise that mimics a header. Skip one block.
            del self._buffer[:BLOCK]
            self._base += BLOCK
            self.rejected_headers += 1
            return True
        name = _member_name(block)
        size = _size(block[124:136])
        typeflag = block[156:157]
        offset = self._base
        del self._buffer[:BLOCK]
        self._base += BLOCK
        self.members_seen += 1
        if typeflag in (b"L", b"K"):
            self._state = "longname"
            self._member = TarMember(name, size, offset, typeflag)
            self._remaining = size
            self._payload = bytearray()
            return True
        if typeflag in (b"x", b"X", b"g"):
            self._state = "pax"
            self._member = TarMember(name, size, offset, typeflag)
            self._remaining = size
            self._payload = bytearray()
            return True
        if self._long_name is not None:
            name = self._long_name
            self._long_name = None
        elif self._pax_name is not None:
            name = self._pax_name
            self._pax_name = None
        if typeflag not in (b"0", b"\0"):
            self._state = "skip"
            self._member = TarMember(name, size, offset, typeflag)
            self._remaining = size
            return True
        if size > MAX_MEMBER_BYTES:
            raise ValueError("original TAR member exceeds the supported size budget")
        self._member = TarMember(name, size, offset, typeflag)
        self._remaining = size
        self._payload = bytearray()
        self._retain = bool(self._predicate(self._member))
        self._state = "data"
        return True

    def _consume_extended(self) -> bool:
        if self._remaining:
            take = min(self._remaining, len(self._buffer))
            if take == 0:
                return False
            self._payload += self._buffer[:take]
            del self._buffer[:take]
            self._base += take
            self._remaining -= take
            if self._remaining:
                return False
        payload = bytes(self._payload)
        self._payload = bytearray()
        if self._state == "longname":
            self._long_name = payload.split(b"\0")[0].decode("utf8", "replace")
        else:
            self._pax_name = _pax_path(payload)
        self._member = None
        self._pad = (512 - (len(payload) % 512)) % 512
        self._state = "padding"
        return True

    def _consume_skip(self) -> bool:
        if self._remaining:
            take = min(self._remaining, len(self._buffer))
            if take == 0:
                return False
            del self._buffer[:take]
            self._base += take
            self._remaining -= take
            if self._remaining:
                return False
        member = self._member
        self._member = None
        self._pad = (512 - (member.size % 512)) % 512 if member is not None else 0
        self._state = "padding"
        return True

    def _consume_data(self) -> bool:
        if self._remaining:
            take = min(self._remaining, len(self._buffer))
            if take == 0:
                return False
            piece = bytes(self._buffer[:take])
            del self._buffer[:take]
            self._base += take
            self._remaining -= take
            if self._retain:
                self._payload += piece
            if self._remaining:
                return False
        member = self._member
        payload = bytes(self._payload)
        self._payload = bytearray()
        self._member = None
        retain = self._retain
        self._retain = False
        if member is not None and retain:
            self.members_emitted += 1
            self._sink(member, payload)
        self._pad = (512 - (member.size % 512)) % 512 if member is not None else 0
        self._state = "padding"
        return True

    def _consume_padding(self) -> bool:
        if self._pad:
            if len(self._buffer) < self._pad:
                return False
            del self._buffer[: self._pad]
            self._base += self._pad
            self._pad = 0
        self._state = "header"
        return True
