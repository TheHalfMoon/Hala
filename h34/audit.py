"""External raw-output audit bound to the independently retained raw SHA-256."""
from __future__ import annotations

import hashlib
import json
import re
import struct
import zlib
from pathlib import Path

from .corpus import build_cases, canonical_cases, case_sha256
from .protocol import _pairs, score_attempt, summarize
from .render import load_font_pin, PILLOW_PIN, RAQM_PIN


def bounded_read(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("bounded regular evidence file required: " + path.name)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("bounded evidence read exceeded")
    return data


def strict_json(data):
    try:
        return json.loads(data, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (RecursionError, OverflowError) as exc:
        raise ValueError("bounded JSON nesting/number invalid") from exc


def verify_png(payload: bytes, width: int, height: int):
    """Decode the pinned renderer's RGB8 noninterlaced PNG envelope and scanlines."""
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("PNG signature invalid")
    offset, kinds, compressed = 8, [], bytearray()
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ValueError("truncated PNG chunk")
        size, = struct.unpack(">I", payload[offset:offset+4])
        kind = payload[offset+4:offset+8]
        end = offset + 12 + size
        if end > len(payload):
            raise ValueError("truncated PNG data")
        data = payload[offset+8:end-4]
        crc, = struct.unpack(">I", payload[end-4:end])
        if zlib.crc32(kind + data) != crc:
            raise ValueError("PNG CRC invalid")
        if kind == b"IHDR":
            if kinds or data != struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0):
                raise ValueError("PNG dimensions or RGB format differ")
        elif kind == b"IDAT":
            if not kinds or kinds[-1] not in (b"IHDR", b"IDAT"):
                raise ValueError("PNG chunk ordering invalid")
            compressed.extend(data)
        elif kind == b"IEND":
            if not kinds or kinds[-1] != b"IDAT" or data or end != len(payload):
                raise ValueError("PNG end invalid")
        else:
            raise ValueError("unexpected PNG renderer chunk")
        kinds.append(kind)
        offset = end
    if not kinds or kinds[-1] != b"IEND":
        raise ValueError("PNG end missing")
    expected = (width * 3 + 1) * height
    decoder = zlib.decompressobj()
    pixels = decoder.decompress(compressed, expected + 1)
    if len(pixels) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("PNG pixel decode invalid or oversized")
    # Every legal PNG filter maps RGB bytes to one well-defined decoded row.
    stride = width * 3 + 1
    if any(pixels[row * stride] > 4 for row in range(height)):
        raise ValueError("PNG scanline filter invalid")


def image_manifest(output, cases):
    path = output / "image_manifest.json"
    if not path.exists():
        return []
    rows = strict_json(bounded_read(path, 65536))
    keys = {"case_id", "path", "sha256", "bytes", "width", "height", "font_sha256",
            "font_revision", "license", "pillow", "raqm", "renderer"}
    if type(rows) is not list or len(rows) > len(cases):
        raise ValueError("invalid bounded image manifest")
    pin = load_font_pin()
    for case, row in zip(cases, rows):
        if type(row) is not dict or set(row) != keys:
            raise ValueError("invalid image manifest schema")
        expected = {"case_id": case["id"], "path": case["id"] + ".png",
                    "width": case["width"], "height": case["height"],
                    "font_sha256": pin["sha256"], "font_revision": pin["upstream_revision"],
                    "license": pin["license"], "pillow": PILLOW_PIN, "raqm": RAQM_PIN,
                    "renderer": "hala-h34-fixed-v1"}
        if any(type(row[k]) is not type(v) or row[k] != v for k, v in expected.items()):
            raise ValueError("image manifest case/renderer binding differs")
        if type(row["bytes"]) is not int or not 0 < row["bytes"] <= 524288:
            raise ValueError("invalid image byte count")
        payload = bounded_read(output / row["path"], 524288)
        if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("image bytes/hash differ from manifest")
        verify_png(payload, case["width"], case["height"])
    return rows


def audit(output: Path, expected_raw_sha256: str) -> dict:
    output = Path(output)
    raw_path = output / "raw_attempts.jsonl"
    if type(expected_raw_sha256) is not str or not re.fullmatch("[a-f0-9]{64}", expected_raw_sha256):
        raise ValueError("independently supplied raw SHA-256 required")
    raw_bytes = bounded_read(raw_path, 1_000_000)
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != expected_raw_sha256:
        raise ValueError("raw output digest differs from independently supplied digest")
    cases = build_cases()
    frozen_path = Path(__file__).parent / "frozen" / "cases.json"
    if frozen_path.read_bytes() != canonical_cases(cases):
        raise ValueError("source frozen case artifact does not match evaluator geometry")
    if bounded_read(output / "cases.sha256", 65) != (case_sha256(cases) + "\n").encode("ascii"):
        raise ValueError("frozen case digest mismatch")
    if raw_bytes and not raw_bytes.endswith(b"\n"):
        raise ValueError("partial raw journal row")
    lines = raw_bytes.decode("utf-8").splitlines()
    if len(lines) > len(cases):
        raise ValueError("too many raw records")
    records = [strict_json(line) for line in lines]
    images = image_manifest(output, cases)
    if len(records) > len(cases) or len(records) > len(images):
        raise ValueError("raw records lack image manifest binding")
    for case, record, image in zip(cases, records, images):
        if type(record) is not dict or set(record) != {"case_id", "language", "raw", "error", "image_sha256"}:
            raise ValueError("invalid strict raw record schema")
        if record["case_id"] != case["id"] or record["language"] != case["language"]:
            raise ValueError("raw records must be exact ordered case prefix with correct language")
        if record["image_sha256"] != image["sha256"]:
            raise ValueError("raw image hash differs from case image")
        raw, error = record["raw"], record["error"]
        if error is None:
            if type(raw) is not str or len(raw.encode("utf-8")) > 2048:
                raise ValueError("raw response type/size invalid")
        elif type(error) is not str or not re.fullmatch("[A-Za-z_][A-Za-z0-9_]{0,79}", error) or raw is not None:
            raise ValueError("invalid error record")
    outcomes = []
    for index, case in enumerate(cases):
        if index >= len(records):
            outcomes.append({"parse_valid": False, "target_hit": False, "wrong_target": False,
                             "abstain": False, "error": True})
        else:
            record = records[index]
            outcomes.append(score_attempt(case, record["raw"], record["error"]))
    return {"raw_sha256": actual_sha, "case_sha256": case_sha256(cases),
            "attempted_records": len(records), "metrics": summarize(cases, outcomes),
            "raw_attempts": records, "complete": len(records) == 32,
            "observations_are_attestation": False, "model_qualified": False}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    args = parser.parse_args()
    result = audit(Path(args.output), args.expected_raw_sha256)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
