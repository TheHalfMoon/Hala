"""H259: verify H250 inline clipboard glyph against pinned MIT Heroicons v1.

Source-only provenance research. Never admits any TRAIN/DEV/OOD data.
No donor bytes or rendered pixels are saved to the Hala repository.
"""
from __future__ import annotations

import argparse
from hashlib import sha1
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree as ET

H250_COMMIT = "009c7880eaa62c4d73b9b0d7b19fa87f8bf71eff"
H250_REMOTE = "https://github.com/Serkanbyx/simple-to-do-list"
HEROICONS_TAG = "v1.0.6"
HEROICONS_ICON_BLOB = "84018bd6c097613289df485ded17149b34862048"
HEROICONS_LICENSE_BLOB = "64c0816664e700d5b552261937493f6e93c2a411"
STATUS = "H259_H250_INLINE_ICON_PROVENANCE_PARTIAL_NOT_ADMITTED"


def require(ok: bool) -> None:
    if not ok:
        raise ValueError("H259_NOT_QUALIFIED_NO_ADMISSION")


def git_blob_sha(blob: bytes) -> str:
    return sha1(b"blob " + str(len(blob)).encode() + b"\x00" + blob).hexdigest()


class SvgReader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.icons: list[dict] = []
        self.current: dict | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "svg":
            require(self.current is None)
            self.current = {"viewbox": a.get("viewbox"), "paths": [],
                            "lines": 0}
        elif self.current is not None and tag == "path":
            require(type(a.get("d")) is str)
            self.current["paths"].append(a["d"])
        elif self.current is not None and tag == "line":
            self.current["lines"] += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "svg":
            require(self.current is not None)
            self.icons.append(self.current)
            self.current = None


def compare_geometry(html: str, official_svg: str) -> tuple[bool, list[str], str]:
    reader = SvgReader()
    reader.feed(html)
    require(reader.current is None and len(reader.icons) == 2
            and reader.icons[0]["lines"] == 2
            and len(reader.icons[0]["paths"]) == 0)
    candidate = reader.icons[1]
    require(candidate["viewbox"] == "0 0 24 24"
            and len(candidate["paths"]) == 2)
    root = ET.fromstring(official_svg)
    require(root.tag.endswith("}svg") and root.get("viewBox") == "0 0 24 24")
    official_paths = [e.get("d") for e in root.iter() if e.tag.endswith("}path")]
    require(len(official_paths) == 1 and type(official_paths[0]) is str)
    first, second = candidate["paths"]
    upstream = official_paths[0]
    # Heroicons v1 uses a relative moveto for its final check-mark subpath;
    # H250 separates that subpath into a second absolute-moveto SVG path.
    m = re.match(r"^M\s*-?\d+(?:\.\d+)?[ ,]+-?\d+(?:\.\d+)?", second)
    require(m is not None and len(second[m.end():]) > 0)
    remainder = upstream[len(first):] if upstream.startswith(first) else ""
    relative_move = remainder[:len(remainder) - len(second[m.end():])] if (
        remainder.endswith(second[m.end():])) else ""
    structural = bool(relative_move and re.fullmatch(
        r"m\s*-?\d+(?:\.\d+)?[ ,]+-?\d+(?:\.\d+)?", relative_move))
    return structural, [first, second], upstream


def raster_difference(paths: list[str], upstream: str) -> dict:
    """Render in-memory in owner Edge; never save or return raster bytes."""
    from playwright.sync_api import sync_playwright
    from xml.sax.saxutils import escape

    def make_svg(items: list[str]) -> str:
        body = "".join('<path d="' + escape(p, {'"': '&quot;'}) + '"/>' for p in items)
        return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
                'fill="none" stroke="#000" stroke-width="2" '
                'stroke-linecap="round" stroke-linejoin="round">' + body + '</svg>')

    with sync_playwright() as pw:
        edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
        opts = {"headless": True}
        if edge.is_file():
            opts["executable_path"] = str(edge)
        else:
            require(Path(pw.chromium.executable_path).is_file())
        browser = pw.chromium.launch(**opts)
        try:
            context = browser.new_context(viewport={"width": 960, "height": 720},
                                          service_workers="block")
            context.set_offline(True)
            try:
                page = context.new_page()
                page.goto("about:blank")
                return page.evaluate("""async ([a, b]) => {
                    async function pixels(svg) {
                        const img = new Image();
                        img.src = 'data:image/svg+xml;charset=utf-8,' +
                                  encodeURIComponent(svg);
                        await img.decode();
                        const canvas = document.createElement('canvas');
                        canvas.width = canvas.height = 192;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0, 192, 192);
                        return ctx.getImageData(0, 0, 192, 192).data;
                    }
                    const left = await pixels(a);
                    const right = await pixels(b);
                    if (left.length !== right.length) throw Error('size');
                    let changed = 0, maxDelta = 0;
                    for (let i = 0; i < left.length; i += 4) {
                        let diff = false;
                        for (let c = 0; c < 4; c++) {
                            const delta = Math.abs(left[i+c] - right[i+c]);
                            maxDelta = Math.max(maxDelta, delta);
                            if (delta !== 0) diff = true;
                        }
                        if (diff) changed++;
                    }
                    return {changed_pixels: changed, max_channel_delta: maxDelta};
                }""", [make_svg(paths), make_svg([upstream])])
            finally:
                context.close()
        finally:
            browser.close()


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, timeout=15, check=True)
    return proc.stdout.strip()


def audit(repo: Path, icon: Path, license_path: Path) -> dict:
    require(repo.is_absolute() and repo.is_dir() and not repo.is_symlink()
            and icon.is_absolute() and license_path.is_absolute()
            and icon.is_file() and license_path.is_file()
            and not icon.is_symlink() and not license_path.is_symlink()
            and icon.stat().st_size <= 4096
            and license_path.stat().st_size <= 4096)
    require(git(repo, "rev-parse", "HEAD") == H250_COMMIT
            and git(repo, "status", "--porcelain") == "")
    remote = git(repo, "remote", "get-url", "origin").removesuffix(".git")
    require(remote == H250_REMOTE)
    html = (repo / "index.html").read_text(encoding="utf-8")
    svg_bytes = icon.read_bytes()
    license_bytes = license_path.read_bytes()
    require(git_blob_sha(svg_bytes) == HEROICONS_ICON_BLOB
            and git_blob_sha(license_bytes) == HEROICONS_LICENSE_BLOB)
    license_text = license_bytes.decode("utf-8")
    require("MIT License" in license_text
            and "Copyright (c) 2020 Refactoring UI Inc." in license_text)
    structural, paths, official = compare_geometry(
        html, svg_bytes.decode("utf-8"))
    require(structural)
    raster = raster_difference(paths, official)
    require(type(raster) is dict
            and type(raster.get("changed_pixels")) is int
            and type(raster.get("max_channel_delta")) is int
            and raster["changed_pixels"] <= 8
            and raster["max_channel_delta"] <= 1)
    app_license = (repo / "LICENSE").read_text(encoding="utf-8")
    require("MIT License" in app_license)
    upstream_notice_in_app_license = "Refactoring UI Inc." in app_license
    return {
        "status": STATUS,
        "original_app_source_pinned": True,
        "official_heroicons_v1_blob_pinned": True,
        "official_heroicons_mit_license_verified": True,
        "candidate_svg_path_structure_matched": structural,
        "normalized_in_memory_raster_changed_pixels": raster["changed_pixels"],
        "normalized_in_memory_raster_max_channel_delta": raster["max_channel_delta"],
        "upstream_notice_in_app_root_license": upstream_notice_in_app_license,
        "third_party_notice_compliance_resolved": False,
        "other_assets_cleared": False,
        "historical_completeness_proven": False,
        "train_admitted": False, "dev_admitted": False, "ood_admitted": False,
        "h173_opened": False, "model_calls": 0, "optimizer_steps": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h250-repo", required=True, type=Path)
    parser.add_argument("--heroicons-svg", required=True, type=Path)
    parser.add_argument("--heroicons-license", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.h250_repo, args.heroicons_svg,
                       args.heroicons_license)
    except (OSError, ValueError, UnicodeError, ET.ParseError,
            subprocess.CalledProcessError, subprocess.TimeoutExpired,
            ImportError, RuntimeError):
        print('{"status":"H259_NOT_QUALIFIED_NO_ADMISSION"}')
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
