"""Publish the H60 domain proof for a domain that has already been ingested.

Only the last shard of a domain can carry the whole-archive digest, because
the digest is only known once the last byte of the archive has been streamed.
The domain proof binds every published shard of that domain to the one
SHA-verified original archive, and it is the receipt that makes the rest of the
corpus usable as training source.

    python -m h60.publish_domain_proof --result <domain-result.json> --staging <dir>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from h60 import ingest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="the domain ingest result JSON")
    parser.add_argument("--staging", required=True)
    args = parser.parse_args()

    token = os.environ["KAGGLE_API_TOKEN"]
    result = json.loads(Path(args.result).read_text("utf8"))
    if not result.get("source_tar_sha256_matches_official"):
        raise SystemExit("refusing to publish a proof for an unverified archive")
    proof = ingest.publish_domain_proof(
        result["source_domain"],
        result,
        result.get("source_window_receipts") or [],
        Path(args.staging),
        token,
    )
    print("H60_DOMAIN_PROOF " + json.dumps(proof, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
