"""H229: independently verify H137 vendor bytes against exact published npm tarballs.

No npm install or execution. Fetch ONLY three fixed public npm registry URLs
in an explicitly opted-in GitHub-hosted read-only job. Never access original
Hala owner media, historical gold, models, or consumed OOD.

Publishing a tarball at a fixed registry URL with correct SHA-1 + registry
SHA-512 does not establish signed npm publisher provenance; only concrete
registry-byte consistency is claimed here.
"""
from __future__ import annotations

from base64 import b64decode
from dataclasses import dataclass
from hashlib import sha1, sha256, sha512
from io import BytesIO
import json
import os
from pathlib import PurePosixPath
import tarfile
from typing import Callable
from urllib.request import Request, urlopen

SCHEMA = "hala.h229.known-three-npm-published-four-assets.v1"
MAX_ARCHIVE_BYTES = 12 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_ASSET_BYTES = 8 * 1024 * 1024
CHUNK = 65536


@dataclass(frozen=True)
class PublishedPackage:
    name: str
    version: str
    shasum: str
    assets: tuple[tuple[str, str], ...]

    @property
    def tarball(self) -> str:
        return (
            "https://registry.npmjs.org/" + self.name + "/-/"
            + self.name + "-" + self.version + ".tgz"
        )

    @property
    def metadata(self) -> str:
        return "https://registry.npmjs.org/" + self.name + "/" + self.version


# npm registry exact-version published tarball SHA-1 read independently,
# 2026-10-30. Only these *public* package tarballs are permitted.
PACKAGES: tuple[PublishedPackage, ...] = (
    PublishedPackage(
        "nouislider", "15.8.1",
        "18729ed29738d2d328a1b5e0429d38eb8be8a696",
        (("dist/nouislider.min.js", "nouislider.min.js"),
         ("dist/nouislider.min.css", "nouislider.min.css")),
    ),
    PublishedPackage(
        "sortablejs", "1.15.6",
        "ff93699493f5b8ab8d828f933227b4988df1d393",
        (("Sortable.min.js", "sortable.min.js"),),
    ),
    PublishedPackage(
        "split.js", "1.6.5",
        "f7f61da1044c9984cb42947df4de4fadb5a3f300",
        (("dist/split.min.js", "split.min.js"),),
    ),
)
ASSET_NAMES = frozenset({
    "nouislider.min.js", "nouislider.min.css", "sortable.min.js",
    "split.min.js",
})


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("H229 original public npm tarball audit blocked: " + message)


def _unique_json(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for k, v in pairs:
        _require(k not in result, "duplicate registry metadata property")
        result[k] = v
    return result


def _deny_constant(_: str) -> None:
    raise ValueError("nonfinite npm registry metadata")


def parse_public_metadata(raw: bytes, package: PublishedPackage) -> str:
    """Return only exact-version npm SHA-512 integrity after validating SHA-1."""
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_METADATA_BYTES,
             "unbounded or empty public version metadata")
    try:
        record = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_json,
            parse_constant=_deny_constant,
        )
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("invalid npm version metadata") from None
    _require(type(record) is dict
             and record.get("name") == package.name
             and record.get("version") == package.version
             and record.get("license") == "MIT",
             "npm exact-version package name/version/license mismatch")
    dist = record.get("dist")
    _require(type(dist) is dict and dist.get("tarball") == package.tarball
             and dist.get("shasum") == package.shasum,
             "published tarball URL or SHA-1 mismatch")
    integrity = dist.get("integrity")
    _require(type(integrity) is str and integrity.startswith("sha512-")
             and len(integrity) < 180, "published SHA-512 SRI missing")
    try:
        expected = b64decode(integrity[len("sha512-"):], validate=True)
    except (ValueError, TypeError):
        raise ValueError("invalid npm SHA-512 SRI encoding") from None
    _require(len(expected) == 64, "published SHA-512 SRI length invalid")
    return integrity


def examine_original_published_archive(
    raw: bytes, package: PublishedPackage, registry_integrity: str,
) -> dict:
    """Bounded in-memory archival inspection. No extraction to disk/execution."""
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_ARCHIVE_BYTES,
             "npm public tarball byte size invalid")
    _require(sha1(raw).hexdigest() == package.shasum,
             "npm published tarball immutable registry SHA-1 mismatch")
    _require(type(registry_integrity) is str
             and registry_integrity.startswith("sha512-"),
             "missing registry SHA-512 attestation")
    try:
        expected = b64decode(registry_integrity[7:], validate=True)
    except (ValueError, TypeError):
        raise ValueError("malformed registry SHA-512 SRI") from None
    _require(len(expected) == 64 and sha512(raw).digest() == expected,
             "npm tarball full SHA-512 integrity mismatch")
    wanted = {
        "package/" + internal: output
        for internal, output in package.assets
    }
    wanted["package/package.json"] = "__package_meta__"
    found: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=BytesIO(raw), mode="r:gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                _require(not path.is_absolute() and ".." not in path.parts,
                         "unsafe tarball member path")
                _require(not (member.issym() or member.islnk()),
                         "published tarball contains linked member")
                if member.name not in wanted:
                    continue
                _require(member.isfile() and member.size <= MAX_ASSET_BYTES
                         and member.size > 0,
                         "public original asset is not a bounded regular file")
                _require(member.name not in found,
                         "duplicate original npm tar entry")
                reader = archive.extractfile(member)
                _require(reader is not None, "missing original npm asset stream")
                blob = reader.read(MAX_ASSET_BYTES + 1)
                _require(len(blob) == member.size,
                         "inconsistent npm asset bytes")
                found[member.name] = blob
    except (tarfile.TarError, OSError, EOFError):
        raise ValueError("invalid published npm tarball") from None
    _require(set(found) == set(wanted), "published npm vendor assets incomplete")
    try:
        package_json = json.loads(
            found["package/package.json"].decode("utf-8"),
            object_pairs_hook=_unique_json, parse_constant=_deny_constant,
        )
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("invalid bundled public package.json") from None
    _require(type(package_json) is dict
             and package_json.get("name") == package.name
             and package_json.get("version") == package.version
             and package_json.get("license") == "MIT",
             "bundled package.json identity/license inconsistent")
    hashes = {
        public_name: sha256(found["package/" + internal]).hexdigest()
        for internal, public_name in package.assets
    }
    _require(set(hashes) == {name for _, name in package.assets},
             "published tarball asset mapping mismatch")
    return {
        "package": package.name,
        "version": package.version,
        "tarball_sha256": sha256(raw).hexdigest(),
        "tarball_sha512_sri": registry_integrity,
        "original_public_asset_sha256": hashes,
        "publisher_attestation_verified": False,
        "owner_h137_files_checked": False,
    }


def _download_exact_public_url(
    url: str, maximum: int, fetch: Callable[[str, int], bytes] | None,
) -> bytes:
    _require(url.startswith("https://registry.npmjs.org/")
             and not any(c in url for c in ("?", "#", "@", "%", "\\"))
             and url.count("://") == 1, "unsupported external URL")
    if fetch is not None:
        raw = fetch(url, maximum)
        _require(type(raw) is bytes and 0 < len(raw) <= maximum,
                 "untrusted public download length")
        return raw
    req = Request(url, headers={
        "Accept": "application/json" if url.endswith(".tgz") is False
        else "application/octet-stream",
        "User-Agent": "Hala-H229-source-only-pinned-registry-audit",
    })
    with urlopen(req, timeout=15) as response:
        _require(response.geturl() == url,
                 "npm registry redirect is not explicitly authorized")
        raw = response.read(maximum + 1)
    _require(0 < len(raw) <= maximum, "public download exceeds bounded size")
    return raw


def verify_all_published_public_assets(
    fetch: Callable[[str, int], bytes] | None = None,
) -> dict:
    """Only three public registry tarballs. No package install or scripts."""
    evidence = []
    observed: dict[str, str] = {}
    for package in PACKAGES:
        meta = _download_exact_public_url(
            package.metadata, MAX_METADATA_BYTES, fetch,
        )
        integrity = parse_public_metadata(meta, package)
        tarball = _download_exact_public_url(
            package.tarball, MAX_ARCHIVE_BYTES, fetch,
        )
        report = examine_original_published_archive(
            tarball, package, integrity,
        )
        _require(not (set(observed) & set(report["original_public_asset_sha256"])),
                 "published vendor asset output name reused")
        observed.update(report["original_public_asset_sha256"])
        evidence.append(report)
    _require(set(observed) == ASSET_NAMES and len(evidence) == 3,
             "incomplete three npm packages or four published assets")
    return {
        "status": "H229_PUBLIC_REGISTRY_THREE_TARBALL_FOUR_ASSETS_VERIFIED",
        "schema": SCHEMA,
        "registry_tarballs_verified": 3,
        "published_original_assets_verified": 4,
        "tarball_receipts": evidence,
        "owner_h137_runtime_verified": False,
        "publisher_signed_provenance_verified": False,
        "official_train_admitted": False,
        "model_calls": 0,
        "optimizer_steps": 0,
        "consumed_original_ood_accessed": 0,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Independent read-only public npm H137 original asset byte hashes"
    )
    parser.add_argument("--fetch-public-registry", action="store_true")
    args = parser.parse_args()
    if not args.fetch_public_registry or os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("H229_REGISTRY_EXPLICIT_HOSTED_ONLY_BLOCKED")
    try:
        receipt = verify_all_published_public_assets()
    except (ValueError, TypeError, OSError, TimeoutError):
        raise SystemExit("H229_PUBLIC_REGISTRY_INDEPENDENT_VERIFICATION_BLOCKED") from None
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
