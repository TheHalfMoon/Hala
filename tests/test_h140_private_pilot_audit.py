"""CPU data integrity guardrails: authentic PNG, source grain and old leakage."""
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from h140.qualification import digest
from h140.private_pilot_audit import read_pilot,historical_exclusions


class H140PrivateSourceAuditTests(unittest.TestCase):
    @staticmethod
    def make_source(folder):
        folder.mkdir(parents=True)
        rows=[]
        for i in range(8):
            image=Image.new("RGB",(960,720),(i*17,40,120))
            raw=io.BytesIO()
            image.save(raw,"PNG")
            filename=f"pilot-{i:02d}.png"
            content=raw.getvalue()
            (folder/filename).write_bytes(content)
            rows.append({
                "id":f"case-{i}","app":"sample-original-app",
                "app_git_sha":"a"*40,"split":"pilot_only_not_training",
                "image":filename,"image_sha256":digest(content),
                "language":"en" if i%2==0 else "ar",
                "start_bbox":{"x":40+i*50,"y":200,"width":25,"height":20},
                "end_bbox":{"x":420+i*50,"y":350,"width":24,"height":22},
                "reference_drag_persisted":True,
                "reference_survived_page_reload":True,
                "model_input_contains_oracle_coordinates":False,
            })
        schema="hala.h140.test-original-source-only.v1"
        doc={"schema":schema,"status":"POSITION_DIVERSE_DIAGNOSTIC_PILOT_ONLY",
             "private":True,"training_eligible":False,"blind_eligible":False,
             "cases":rows,"model_neural_calls":0}
        (folder/"manifest.json").write_text(json.dumps(doc),encoding="utf8")
        policy={"sha":"a"*40,"source":"sample-original-app","count":8,
                "schema":schema,"start":8,"end":8}
        return doc,policy

    def test_valid_real_png_and_reference_actions_admitted(self):
        with tempfile.TemporaryDirectory() as base:
            root=Path(base)/"test-pilot"
            _,policy=self.make_source(root)
            result=read_pilot(Path(base),"test-pilot",policy)
            self.assertEqual(result["case_count"],8)
            self.assertEqual(result["languages"],{"en":4,"ar":4})
            self.assertEqual(result["start_bins"],8)
            self.assertEqual(result["end_bins"],8)

    def test_invalid_png_rejected_even_if_sha_and_size_plausible(self):
        with tempfile.TemporaryDirectory() as base:
            folder=Path(base)/"test-pilot"
            doc,policy=self.make_source(folder)
            fake=(b"\x89PNG\r\n\x1a\n"+b"\x00\x00\x00\x0dIHDR"+
                  (960).to_bytes(4,"big")+(720).to_bytes(4,"big"))
            (folder/"pilot-00.png").write_bytes(fake)
            doc["cases"][0]["image_sha256"]=digest(fake)
            (folder/"manifest.json").write_text(json.dumps(doc),encoding="utf8")
            with self.assertRaises(Exception):
                read_pilot(Path(base),"test-pilot",policy)

    def test_oracle_leakage_or_false_reload_persistence_rejected(self):
        with tempfile.TemporaryDirectory() as base:
            folder=Path(base)/"test-pilot"
            doc,policy=self.make_source(folder)
            doc["cases"][0]["reference_survived_page_reload"]=False
            (folder/"manifest.json").write_text(json.dumps(doc),encoding="utf8")
            with self.assertRaisesRegex(ValueError,"receipt invalid"):
                read_pilot(Path(base),"test-pilot",policy)

    def test_history_is_fail_closed_when_past_data_missing(self):
        with tempfile.TemporaryDirectory() as base:
            with self.assertRaisesRegex(ValueError,"historical evaluated stage source missing"):
                historical_exclusions(base)


if __name__=="__main__":
    unittest.main()
