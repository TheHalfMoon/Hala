"""H275 original author-owned GUI source contracts; no browser or dataset use."""
from html.parser import HTMLParser
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"h275"/"queueboard.html"

class Tags(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags=[]
    def handle_starttag(self,tag,attrs):
        self.tags.append((tag,dict(attrs)))

class OriginalQueueboardSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html=SOURCE.read_text("utf-8")
        cls.parser=Tags()
        cls.parser.feed(cls.html)

    def test_offline_original_source_and_no_external_media(self):
        h=self.html
        self.assertIn("SPDX-License-Identifier: Apache-2.0",h)
        self.assertIn('data-source-family="hala-original-queueboard-h275"',h)
        self.assertIn("Original synthetic service desk",h)
        self.assertIn('connect-src \'none\'',h)
        self.assertIn('default-src \'none\'',h)
        self.assertIn('form-action \'none\'',h)
        for forbidden in (
            "<img","<link","<iframe","<audio","<video",
            "src=","fetch(","XMLHttpRequest","WebSocket","import(",
            "localStorage","sessionStorage","document.cookie",
            "Math.random","Date.now","location.href","window.open",
            "https://","http://","stockroom.html",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden,h)
        self.assertEqual(len([x for x,_ in self.parser.tags if x=="script"]),1)

    def test_six_fictional_rows_not_personal_or_clinical_data(self):
        h=self.html
        ids=re.findall(r'\{id:"(Q-\d{3})"',h)
        self.assertEqual(ids,["Q-104","Q-118","Q-126","Q-137","Q-142","Q-153"])
        self.assertEqual(len(set(ids)),6)
        self.assertIn('const SEED=Object.freeze([',h)
        self.assertIn("no real customers or accounts",h)
        self.assertNotIn("patient",h.lower())
        self.assertNotIn("diagnosis",h.lower())
        self.assertIn("h275-original-queueboard",h)

    def test_native_button_actions_have_independent_state_oracle(self):
        h=self.html
        for item in (
            'action.dataset.action="complete"',
            'action.dataset.ticketId=ticket.id',
            'row.dataset.status=ticket.done?"done":"waiting"',
            'ticket.done=true;',
            'byId("completed-count").textContent=String(',
            'byId("urgent-count").textContent=String(',
            'action.disabled=ticket.done',
            'const row=document.createElement("article")',
            'list.replaceChildren()',
            'render();',
        ):
            with self.subTest(item=item):
                self.assertIn(item,h)
        for id in ("lang","reset","priority","all","search","completed-count","ticket-list"):
            self.assertEqual(len([
                1 for tag,attrs in self.parser.tags if attrs.get("id")==id]),1)

    def test_bilingual_rtl_filter_and_screenshot_design(self):
        h=self.html
        self.assertIn('document.documentElement.dir=language==="ar"?"rtl":"ltr"',h)
        self.assertIn('byId("search").blur()',h)
        self.assertIn('byId("search").addEventListener("input"',h)
        self.assertIn('byId("priority").addEventListener("click"',h)
        self.assertIn("الطلب",h)
        self.assertIn("مكتمل",h)
        self.assertIn("استلام المستندات",h)
        self.assertIn("min-height:54px",h)
        self.assertIn("width:min(960px,100%)",h)
        self.assertNotIn("@keyframes",h)
        self.assertNotIn("animation:",h)
        self.assertNotIn("setInterval(",h)
        self.assertNotIn("setTimeout(",h)

if __name__=="__main__":
    unittest.main()
