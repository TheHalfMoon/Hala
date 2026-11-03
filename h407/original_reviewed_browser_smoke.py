"""H407 actual original first-party local browser native click-review smoke.

Two synthetic HTML fixtures with 1 vs 2 visible red controls. A source-known
injected callback (NOT a model or independent reviewer) proposes a click.
The separate source-owned fixture reviewer confirms 1 vs 2 matches bound
to the screenshot/task/action. Test private local only, no network egress,
private data, media files, training, or external OOD.
"""
from __future__ import annotations
from hashlib import sha256
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from threading import Thread

from h201.original_native_action_executor import exact_original_loopback_url
from h407.reviewed_action_bridge import (
    VisualTargetReview,action_digest,run_source_reviewed_policy,
)

TASK="Click the unique visible red square."


def source_html(duplicates: bool) -> bytes:
    second=('<button id="other" style="position:absolute;left:390px;top:150px;'
            'width:40px;height:40px;background:red">B</button>'
            if duplicates else "")
    html=(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<style>html,body{margin:0;padding:0;background:#eee}"
        "</style></head><body>"
        '<button id="target" style="position:absolute;left:180px;top:150px;'
        'width:40px;height:40px;background:red"'
        " onclick=\"document.getElementById('counter').textContent='1'\">A</button>"
        +second+
        '<div id="counter" style="position:absolute;top:300px">0</div>'
        "</body></html>"
    )
    return html.encode("utf8")


def original_once(channel: str, duplicates: bool) -> dict:
    if channel not in ("msedge","chromium"):
        raise ValueError("H407 requires owner-installed first-party local browser")
    html=source_html(duplicates)
    class OnlyLocalHTML(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!="/snap/index.html":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Content-Length",str(len(html)))
            self.send_header("Cache-Control","no-store")
            self.send_header("Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; connect-src 'none'")
            self.end_headers()
            self.wfile.write(html)
        def log_message(self,*_args):
            return

    server=ThreadingHTTPServer(("127.0.0.1",0),OnlyLocalHTML)
    thread=Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        from playwright.sync_api import sync_playwright
        from h174.pickr_native_smoke import allow_loopback_request
        port=server.server_address[1]
        url=exact_original_loopback_url(port)
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(channel=channel,headless=True)
            try:
                context=browser.new_context(
                    viewport={"width":960,"height":720},
                    device_scale_factor=1,reduced_motion="reduce",
                    service_workers="block",accept_downloads=False)
                try:
                    context.route("**/*",lambda route:route.continue_()
                        if allow_loopback_request(route.request.url,port)
                        else route.abort())
                    context.route_web_socket("**/*",lambda ws:ws.close())
                    page=context.new_page()
                    response=page.goto(url,wait_until="networkidle",
                                       timeout=20000)
                    if response is None or response.status!=200:
                        raise ValueError("H407 local original source browser not loaded")
                    page.locator("#target").wait_for(state="visible")
                    page.wait_for_timeout(100)
                    predictor_calls=[]
                    reviewer_calls=[]
                    def synthetic_fixture_predictor(png,prompt):
                        predictor_calls.append((sha256(png).hexdigest(),prompt))
                        return json.dumps({"actions":[{"kind":"click",
                            "point":[200,170]}]},separators=(",",":"))
                    def source_owned_reviewer(png,task,action):
                        # Only fixture-owned source count, NOT model judgement.
                        reviewer_calls.append(1)
                        return VisualTargetReview(
                            png_sha256=sha256(png).hexdigest(),
                            task_sha256=sha256(task.encode("utf8")).hexdigest(),
                            action_sha256=action_digest(action),
                            matching_targets=2 if duplicates else 1,
                            visible_box=(180,150,220,190))
                    if duplicates:
                        try:
                            run_source_reviewed_policy(
                                page,TASK,url,synthetic_fixture_predictor,
                                source_owned_reviewer)
                        except ValueError as exc:
                            if "unique" not in str(exc):
                                raise
                        else:
                            raise AssertionError("H407 duplicate target wrongly clicked")
                        if page.locator("#counter").inner_text()!="0":
                            raise AssertionError("H407 duplicated target emitted browser action")
                        if len(predictor_calls)!=1 or len(reviewer_calls)!=1:
                            raise AssertionError("H407 bounded model/review callback count")
                        return {"duplicates":2,"native_events":0,
                            "status":"REVIEW_REFUSED_AMBIGUOUS_NO_BROWSER_INPUT"}
                    result=run_source_reviewed_policy(
                        page,TASK,url,synthetic_fixture_predictor,
                        source_owned_reviewer)
                    if (page.locator("#counter").inner_text()!="1"
                        or result["native_events"]!=1
                        or not result["preaction_png_verified"]
                        or not result["local_review_bound_to_png"]
                        or len(predictor_calls)!=1
                        or len(reviewer_calls)!=1):
                        raise AssertionError("H407 local source review action not verified")
                    return {"duplicates":1,"native_events":1,
                        "status":"SOURCE_KNOWN_REVIEWED_FIXTURE_INPUT_ONLY"}
                finally:
                    context.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def qualify_local_original(channel="msedge"):
    one=original_once(channel,False)
    two=original_once(channel,True)
    if one["native_events"]!=1 or two["native_events"]!=0:
        raise ValueError("H407 local browser action safety boundary failed")
    return {"status":"H407_FIRST_PARTY_FIXTURE_REVIEW_BOUNDARY_PASS",
        "actual_browser_channel":channel,"source_only":True,
        "first_party_local_cases":2,"reviewed_unique_source_actions":1,
        "ambiguous_source_actions_refused":1,
        "real_hala_model_used":False,"externally_authenticated_reviewer":False,
        "independent_ood":False,"original_gui_train_approved":False,
        "model_release_qualified":False}


if __name__=="__main__":
    print(json.dumps(qualify_local_original(),sort_keys=True))
