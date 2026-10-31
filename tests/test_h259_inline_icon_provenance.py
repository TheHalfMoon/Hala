"""H259 invented SVG-only regression; no original app, icon, screenshots or gold."""
from __future__ import annotations

from hashlib import sha1
import pytest

from h259.inline_icon_provenance import (
    H250_COMMIT, HEROICONS_ICON_BLOB, HEROICONS_LICENSE_BLOB,
    compare_geometry, git_blob_sha, require,
)


def html(first="M1 1L2 2", second="M3 4l1 1", viewbox="0 0 24 24"):
    return (
        '<html><body><svg viewBox="0 0 24 24">'
        '<line x1="0" y1="0" x2="1" y2="1"/>'
        '<line x1="1" y1="1" x2="2" y2="2"/>'
        '</svg><svg viewBox="' + viewbox + '"><path d="' + first +
        '"/><path d="' + second + '"/></svg></body></html>'
    )


def official(d="M1 1L2 2m1 2l1 1"):
    return ('<svg xmlns="http://www.w3.org/2000/svg" '
            'viewBox="0 0 24 24"><path d="' + d + '"/></svg>')


def test_invented_split_relative_moveto_structure():
    match, paths, upstream = compare_geometry(html(), official())
    assert match is True
    assert len(paths) == 2
    assert upstream.endswith("l1 1")


def test_modified_upstream_path_is_not_match():
    match, _, _ = compare_geometry(html(), official("M1 1L2 3m1 2l1 1"))
    assert match is False


def test_modified_second_subpath_is_not_match():
    match, _, _ = compare_geometry(html(second="M3 4l2 2"), official())
    assert match is False


def test_wrong_viewbox_rejected():
    with pytest.raises(ValueError, match="H259_NOT_QUALIFIED"):
        compare_geometry(html(viewbox="0 0 48 48"), official())


def test_unexpected_svg_count_rejected():
    with pytest.raises(ValueError, match="H259_NOT_QUALIFIED"):
        compare_geometry(html() + '<svg viewBox="0 0 24 24"></svg>', official())


def test_unexpected_extra_upstream_path_rejected():
    with pytest.raises(ValueError, match="H259_NOT_QUALIFIED"):
        compare_geometry(html(), official().replace("</svg>",
            '<path d="M0 0L1 1"/></svg>'))


def test_malformed_svg_rejected():
    with pytest.raises(Exception):
        compare_geometry(html(), "<svg")


def test_git_blob_digest_is_real_git_blob_formula():
    blob = b"synthetic invented test"
    expected = sha1(b"blob " + str(len(blob)).encode() + b"\x00" + blob).hexdigest()
    assert git_blob_sha(blob) == expected


def test_upstream_refs_are_fixed_and_not_private():
    assert len(H250_COMMIT) == len(HEROICONS_ICON_BLOB) == len(HEROICONS_LICENSE_BLOB) == 40
    with pytest.raises(ValueError):
        require(False)
