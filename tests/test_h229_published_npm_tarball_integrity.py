"""H229 pure stdlib tests against synthetic tarballs; no network or owner data."""
from __future__ import annotations

from base64 import b64encode
from hashlib import sha1, sha256, sha512
from io import BytesIO
import json
import tarfile
import unittest
from unittest.mock import patch

from h229.verified_published_npm_assets import (
    ASSET_NAMES, PACKAGES, PublishedPackage, SCHEMA,
    examine_original_published_archive, parse_public_metadata,
    verify_all_published_public_assets,
)


def sample_archive(assets, name="fake-public", version="1.0.0",
                   license="MIT", extra=None):
    payload = {"package/package.json": json.dumps({
        "name": name, "version": version, "license": license,
    }).encode()}
    payload.update({"package/"+path: data for path, data in assets.items()})
    payload.update(extra or {})
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tf:
        for path, content in sorted(payload.items()):
            item = tarfile.TarInfo(path)
            item.size = len(content)
            tf.addfile(item, BytesIO(content))
    return buf.getvalue()


def fixture(assets=None, extra=None, license="MIT"):
    parts = {"dist/demo.min.js": b"original public safe synthetic package file"}
    if assets is not None:
        parts = assets
    blob = sample_archive(parts, extra=extra, license=license)
    pkg = PublishedPackage("fake-public", "1.0.0", sha1(blob).hexdigest(),
                           (("dist/demo.min.js", "demo.min.js"),))
    integrity = "sha512-" + b64encode(sha512(blob).digest()).decode()
    record = {
        "name": pkg.name, "version": pkg.version, "license": "MIT",
        "dist": {
            "tarball": pkg.tarball, "shasum": pkg.shasum,
            "integrity": integrity,
        },
    }
    return pkg, blob, integrity, record


class PublishedNpmArtifactVerificationTests(unittest.TestCase):
    def test_genuine_synthetic_archive_has_exact_sha512_sha1_and_file_hash(self):
        pkg, blob, sri, record = fixture()
        self.assertEqual(parse_public_metadata(json.dumps(record).encode(), pkg), sri)
        result = examine_original_published_archive(blob, pkg, sri)
        self.assertEqual(result["original_public_asset_sha256"]["demo.min.js"],
                         sha256(b"original public safe synthetic package file").hexdigest())
        self.assertFalse(result["publisher_attestation_verified"])
        self.assertFalse(result["owner_h137_files_checked"])

    def test_exact_three_public_packages_and_four_original_asset_names(self):
        self.assertEqual(len(PACKAGES), 3)
        self.assertEqual(set(ASSET_NAMES), {
            "nouislider.min.js", "nouislider.min.css",
            "sortable.min.js", "split.min.js",
        })
        self.assertEqual({(x.name, x.version) for x in PACKAGES}, {
            ("nouislider","15.8.1"),("sortablejs","1.15.6"),
            ("split.js","1.6.5"),
        })
        self.assertEqual(len({x.shasum for x in PACKAGES}), 3)
        self.assertEqual(len({n for x in PACKAGES for _, n in x.assets}), 4)
        self.assertEqual(SCHEMA, "hala.h229.known-three-npm-published-four-assets.v1")

    def test_tarball_byte_mutation_is_never_accepted(self):
        pkg, blob, sri, _ = fixture()
        corrupt = blob[:12] + bytes([blob[12] ^ 1]) + blob[13:]
        with self.assertRaisesRegex(ValueError, "SHA-1 mismatch"):
            examine_original_published_archive(corrupt, pkg, sri)

    def test_registry_integrity_mismatch_rejected_even_with_sha1_correct(self):
        pkg, blob, sri, _ = fixture()
        wrong = "sha512-" + b64encode(b"1"*64).decode()
        with self.assertRaisesRegex(ValueError, "SHA-512"):
            examine_original_published_archive(blob, pkg, wrong)

    def test_registry_metadata_wrong_url_license_or_shasum_denied(self):
        pkg, _, _, source = fixture()
        for changed in (
            ("name", "typosquat"), ("version", "2.0.0"),
            ("license", "GPL"), ("dist", dict(source["dist"], shasum="1"*40)),
            ("dist", dict(source["dist"], tarball="https://example.com/alien.tgz")),
            ("dist", dict(source["dist"], integrity="sha256-AAAA")),
        ):
            key, value = changed
            with self.subTest(field=key, value=value):
                row = dict(source)
                row[key] = value
                with self.assertRaises(ValueError):
                    parse_public_metadata(json.dumps(row).encode(), pkg)

    def test_bad_json_duplicate_keys_and_nonfinite_constants(self):
        pkg, _, _, _ = fixture()
        with self.assertRaises(ValueError):
            parse_public_metadata(b'{"name":"a","name":"b"}', pkg)
        with self.assertRaises(ValueError):
            parse_public_metadata(b'{"dist":NaN}', pkg)

    def test_missing_original_js_file_denied(self):
        pkg, blob, sri, _ = fixture(assets={"dist/other.js": b"wrong"})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            examine_original_published_archive(blob, pkg, sri)

    def test_swapped_npm_package_identity_denied(self):
        pkg, blob, sri, _ = fixture()
        wrong = PublishedPackage("other-package", "1.0.0", pkg.shasum,
                                 pkg.assets)
        with self.assertRaisesRegex(ValueError, "package.json identity"):
            examine_original_published_archive(blob, wrong, sri)

    def test_linked_artifact_member_denied(self):
        pkg, _, _, _ = fixture()
        buf=BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            content=json.dumps({"name":pkg.name,"version":pkg.version,
                                "license":"MIT"}).encode()
            m=tarfile.TarInfo("package/package.json")
            m.size=len(content)
            tf.addfile(m, BytesIO(content))
            link=tarfile.TarInfo("package/dist/demo.min.js")
            link.type=tarfile.SYMTYPE
            link.linkname="../outside"
            tf.addfile(link)
        blob=buf.getvalue()
        new_pkg=PublishedPackage(pkg.name,pkg.version,sha1(blob).hexdigest(),pkg.assets)
        sri="sha512-"+b64encode(sha512(blob).digest()).decode()
        with self.assertRaisesRegex(ValueError, "linked member"):
            examine_original_published_archive(blob,new_pkg,sri)

    def test_registry_source_is_only_injected_while_unit_testing(self):
        # Avoid real public network in CI source-only negative tests.
        values=[]
        fixtures=[]
        expected_assets={}
        expected_tars={}
        expected_sris={}
        for i in range(3):
            name=f"synthetic-public-{i}"
            path=f"dist/original-{i}.js"
            blob=sample_archive({path:f"original-{i}".encode()},
                                name=name,version="1.0.0")
            pkg=PublishedPackage(name,"1.0.0",sha1(blob).hexdigest(),
                                 ((path,f"output-{i}.js"),))
            sri="sha512-"+b64encode(sha512(blob).digest()).decode()
            meta=json.dumps({
                "name":name,"version":"1.0.0","license":"MIT",
                "dist":{"tarball":pkg.tarball,
                        "shasum":pkg.shasum,"integrity":sri}
            }).encode()
            values.append(pkg)
            expected_assets[f"output-{i}.js"]=sha256(f"original-{i}".encode()).hexdigest()
            expected_tars[name]=sha256(blob).hexdigest()
            expected_sris[name]=sri
            fixtures.append((pkg.metadata,meta))
            fixtures.append((pkg.tarball,blob))
        receipt_map=dict(fixtures)
        with patch("h229.verified_published_npm_assets.PACKAGES",tuple(values)),\
             patch("h229.verified_published_npm_assets.ASSET_NAMES",
                   frozenset(f"output-{i}.js" for i in range(3))),\
             patch("h229.verified_published_npm_assets.PINNED_PUBLISHED_ORIGINAL_SHA256",
                   expected_assets),\
             patch("h229.verified_published_npm_assets.PINNED_PUBLISHED_TARBALL_SHA256",
                   expected_tars),\
             patch("h229.verified_published_npm_assets.PINNED_REGISTRY_SRI",
                   expected_sris):
            result=verify_all_published_public_assets(
                lambda url, limit: receipt_map[url])
        self.assertEqual(result["registry_tarballs_verified"],3)
        self.assertEqual(result["published_original_assets_verified"],4)
        self.assertFalse(result["official_train_admitted"])
        self.assertEqual(result["model_calls"],0)
        self.assertEqual(result["optimizer_steps"],0)
        self.assertEqual(result["consumed_original_ood_accessed"],0)


if __name__=="__main__":
    unittest.main()
