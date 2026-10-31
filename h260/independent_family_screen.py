"""H260: two independent original-app source families, owner-only research screening.

No training admission, private media, raw predictions, or scored OOD access.
"""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess

FAMILIES = {
    "dacheson": ("https://github.com/dacheson/pomodoro-timer",
                 "#timer-text", ".mode-tab[data-mode='shortBreak']",
                 "#start-btn", "#pause-btn"),
    "dinnocenzo": ("https://github.com/dinnocenzo/pomodoro-timer-app",
                   "#timer-display", ".session-btn[data-type='shortBreak']",
                   "#btn-start", "#btn-pause"),
}
STATUS = "H260_INDEPENDENT_FAMILY_RESEARCH_ONLY_NOT_ADMITTED"
MEDIA = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".svg",
         ".woff", ".woff2", ".ttf", ".otf", ".mp3", ".wav"}
SOURCE = {".html", ".css", ".js"}
URL = re.compile(r"https?://[^\s'\"<>\)]+", re.I)


def require(ok: bool) -> None:
    if not ok:
        raise ValueError("H260_NOT_QUALIFIED_NO_ADMISSION")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, check=True, timeout=45).stdout.strip()


class MediaTags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.remote = 0
        self.media = 0
        self.inline_svg = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "svg":
            self.inline_svg += 1
        if tag in {"img", "picture", "video", "audio", "source", "iframe", "image", "object", "embed"}:
            self.media += 1
        for k in ("src", "href", "srcset"):
            v = a.get(k) or ""
            if v.startswith(("http:", "https:", "//", "data:")):
                self.remote += 1


def scan_invented_text(html: str, css: str, js: str) -> dict:
    """Pure scan, suitable for invented synthetic regression fixtures."""
    tags = MediaTags()
    tags.feed(html)
    text = "\n".join((html, css, js))
    return {
        "external_url_references": len(set(URL.findall(text))) + tags.remote
        + len(re.findall(r"(?<!:)//[a-z0-9.-]+/", css, re.I))
        + len(re.findall(r"@import\b", css, re.I)),
        "css_url_references": len(re.findall(r"url\s*\(", css, re.I)),
        "media_tags": tags.media,
        "inline_svg_tags": tags.inline_svg,
    }


def source_screen(repo: Path, family: str, expected_commit: str) -> dict:
    require(family in FAMILIES and repo.is_absolute() and repo.is_dir()
            and not repo.is_symlink() and re.fullmatch(r"[0-9a-f]{40}", expected_commit)
            is not None)
    require(git(repo, "status", "--porcelain") == ""
            and git(repo, "rev-parse", "HEAD") == expected_commit)
    require(git(repo, "remote", "get-url", "origin").removesuffix(".git")
            == FAMILIES[family][0])
    tracked = git(repo, "ls-files").splitlines()
    require("LICENSE" in tracked and "index.html" in tracked
            and len(tracked) <= 30)
    license_text = (repo / "LICENSE").read_text(encoding="utf-8")
    require("MIT License" in license_text and "Copyright" in license_text)
    histories = git(repo, "rev-list", "--all").splitlines()
    require(1 <= len(histories) <= 100)
    historical = set()
    for commit in histories:
        historical.update(git(repo, "ls-tree", "-r", "--name-only", commit).splitlines())
    sources = [x for x in tracked if Path(x).suffix.lower() in SOURCE]
    require(len(sources) >= 3 and len(sources) <= 10)
    contents = {}
    for name in sources:
        path = repo / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 150000)
        contents[name] = path.read_text(encoding="utf-8")
    html = contents["index.html"]
    scan = scan_invented_text(html,
                             "\n".join(v for n, v in contents.items() if n.endswith(".css")),
                             "\n".join(v for n, v in contents.items() if n.endswith(".js")))
    require(scan["external_url_references"] == 0
            and scan["css_url_references"] == 0 and scan["media_tags"] == 0)
    media = [p for p in historical if Path(p).suffix.lower() in MEDIA]
    # Historical media remains quarantined even when not loaded in current GUI.
    return {"historical_local_commit_count": len(histories),
            "historical_media_path_count": len(media),
            "inline_svg_tag_count": scan["inline_svg_tags"],
            "static_external_references": scan["external_url_references"],
            "history_complete_globally": False,
            "all_media_rights_proven": False}


def native_screen(repo: Path, family: str) -> dict:
    from playwright.sync_api import sync_playwright
    _, display, mode, start, pause = FAMILIES[family]
    with sync_playwright() as pw:
        edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
        require(edge.is_file())
        browser = pw.chromium.launch(executable_path=str(edge), headless=True)
        try:
            context = browser.new_context(viewport={"width": 960, "height": 720},
                                          service_workers="block")
            context.set_offline(True)
            try:
                page = context.new_page()
                requests: list[str] = []
                errors: list[str] = []
                page.on("request", lambda r: requests.append(r.url)
                        if r.url.startswith(("http:", "https:")) else None)
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto((repo / "index.html").as_uri(), wait_until="load")
                initial = page.locator(display).inner_text() == "25:00"
                page.locator(mode).click()
                changed = page.locator(display).inner_text() == "05:00"
                page.locator(start).click()
                started = (page.locator(pause).is_visible() if family == "dacheson"
                           else page.locator(pause).is_enabled())
                page.locator(pause).click()
                paused = (page.locator(start).is_visible() if family == "dacheson"
                          else page.locator(start).is_enabled())
                task = True
                if family == "dinnocenzo":
                    page.locator("#it").fill("Invented H260 task")
                    page.locator("#bAdd").click()
                    task = "Invented H260 task" in page.locator("#tasks").inner_text()
                require(initial and changed and started and paused and task
                        and not requests and not errors and page.locator("img").count() == 0)
                return {"native_state_checks_passed": 5 if family == "dinnocenzo" else 4,
                        "external_http_requests": 0, "js_errors": 0,
                        "screenshots_saved": 0}
            finally:
                context.close()
        finally:
            browser.close()


def audit(repo: Path, family: str, expected_commit: str) -> dict:
    static = source_screen(repo, family, expected_commit)
    native = native_screen(repo, family)
    return {"status": STATUS, "family": family, **static, **native,
            "source_family_disjointness_globally_proven": False,
            "train_admitted": False, "dev_admitted": False, "ood_admitted": False,
            "h173_opened": False, "model_calls": 0, "optimizer_steps": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    try:
        result = audit(args.repo, args.family, args.expected_commit)
    except (OSError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, ImportError, RuntimeError, KeyError):
        print('{"status":"H260_NOT_QUALIFIED_NO_ADMISSION"}')
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
