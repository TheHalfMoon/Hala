"""H59 source-full million-row MANIFEST tests only; no paid GPU or 61GB fetch."""
import hashlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from h59 import master_data_plan as m
from h59 import fetch_window as f
from h59 import verify_full_source as v

SOURCE=Path(__file__).resolve().parents[1]/"h59/full_source_windows.json"

class OfflineResponse(io.BytesIO):
    def __init__(self,content,status=206,range_header=""):
        super().__init__(content)
        self.status=status
        self.headers={"Content-Range":range_header,"Content-Length":str(len(content))}
    def __enter__(self):return self
    def __exit__(self,*a):self.close()

class MillionRowSourceMapTests(unittest.TestCase):
    def test_original_four_train_domains_cover_exact_hf_source(self):
        actual=json.loads(SOURCE.read_text("utf8"))
        r=m.check_plan(actual)
        self.assertEqual(r["upstream_tasks"],3560142)
        self.assertEqual(r["upstream_images"],286012)
        self.assertEqual(r["planned_bytes"],61195154432)
        self.assertEqual(r["planned_windows"],116)
        self.assertFalse(r["source_train_materialized"])
        self.assertFalse(r["neural_training_executed"])
        for name,info in m.DOMAINS.items():
            self.assertEqual(len(info["upstream_lfs_sha256"]),64)
            self.assertEqual(actual["source_domains"][name]["upstream_lfs_sha256"],info["upstream_lfs_sha256"])

    def test_each_fixed_train_byte_in_single_nonoverlapping_aligned_window(self):
        d=m.build()
        for domain,expected in m.DOMAINS.items():
            windows=[x for x in d["windows"] if x["source_domain"]==domain]
            cursor=0
            for x in windows:
                self.assertEqual(x["byte_start"],cursor)
                self.assertEqual(x["byte_end_inclusive"]+1,x["byte_start"]+x["expected_length"])
                self.assertEqual(x["expected_length"]%512,0)
                self.assertLessEqual(x["expected_length"],512*1024*1024)
                self.assertFalse(x["window_downloaded"])
                self.assertFalse(x["provider_private_storage_verified"])
                self.assertIsNone(x["source_window_sha256"])
                self.assertEqual(x["reusable_train_examples_ready"],0)
                self.assertIn("/train/",x["url"])
                self.assertNotIn("/eval/",x["url"])
                cursor+=x["expected_length"]
            self.assertEqual(cursor,expected["bytes"])

    def test_catalog_is_not_false_claim_of_millions_saved_or_trained(self):
        x=json.loads(SOURCE.read_text("utf8"))
        self.assertEqual(x["fully_materialized_original_drag_tasks"],0)
        self.assertEqual(x["fully_indexed_unique_original_images"],0)
        self.assertFalse(x["full_private_kaggle_uploaded"])
        self.assertEqual(x["train_gpu_steps"],0)
        self.assertFalse(x["model_checkpoint_trained"])
        self.assertEqual(x["existing_source_qualified_private_kaggle"]["original_hcompany_train_drag_annotations"],2346)

    def test_silent_source_range_mutation_fails(self):
        x=m.build()
        x["windows"][0]["byte_end_inclusive"]+=512
        with self.assertRaises(ValueError):m.check_plan(x)
        x=m.build();x["source_revision"]="0"*40
        with self.assertRaises(ValueError):m.check_plan(x)

    def tiny_source(self,folder,data=b"x"*1024):
        p=Path(folder)/"mini.json"
        window=dict(id="h59-sheet-0000",source_domain="sheet",url="https://example.invalid/train/sheet.tar",
            byte_start=0,byte_end_inclusive=len(data)-1,expected_length=len(data),
            total_tar_bytes=len(data),
            expected_content_range=f"bytes 0-{len(data)-1}/{len(data)}")
        plan=dict(source_repo=m.REPO,source_revision=m.REV,windows=[window])
        p.write_text(json.dumps(plan),encoding="utf8")
        return p,window,data

    def test_no_network_and_invalid_window_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                f.get_window(SOURCE,"h59-sheet-0000",t,allow_network=False)
            with self.assertRaises(ValueError):
                f.get_window(SOURCE,"h59-unknown-9999",t,allow_network=True)

    def test_mocked_exact_206_stream_creates_sha_but_not_private_claim(self):
        with tempfile.TemporaryDirectory() as t:
            plan,record,data=self.tiny_source(t)
            response=OfflineResponse(data,range_header=record["expected_content_range"])
            with patch.object(f,"check_plan",return_value=None),patch.object(f.urllib.request,"urlopen",return_value=response) as network:
                v=f.get_window(plan,record["id"],Path(t)/"download",allow_network=True)
            self.assertEqual(network.call_count,1)
            self.assertEqual(v["bytes_verified"],1024)
            self.assertEqual(v["original_window_sha256"],hashlib.sha256(data).hexdigest())
            self.assertFalse(v["provider_private_kaggle_uploaded"])
            self.assertFalse(v["provider_private_kaggle_readback_sha_verified"])
            self.assertEqual(v["gpu_training_steps"],0)
            with self.assertRaises(FileExistsError):
                with patch.object(f,"check_plan",return_value=None):
                    f.get_window(plan,record["id"],Path(t)/"download",allow_network=True)

    def test_http_200_refused_as_range_failure(self):
        with tempfile.TemporaryDirectory() as t:
            plan,record,data=self.tiny_source(t)
            response=OfflineResponse(data,status=200,range_header=record["expected_content_range"])
            with patch.object(f,"check_plan",return_value=None),patch.object(f.urllib.request,"urlopen",return_value=response):
                with self.assertRaises(ValueError):
                    f.get_window(plan,record["id"],Path(t)/"out",allow_network=True)
            self.assertFalse(list((Path(t)/"out").glob("*.receipt.json")))

    def test_incomplete_download_can_resume_only_missing_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            plan,record,data=self.tiny_source(t)
            target=Path(t)/"windows"
            target.mkdir()
            (target/(record["id"]+".part")).write_bytes(data[:128])
            response=OfflineResponse(data[128:],range_header="bytes 128-1023/1024")
            with patch.object(f,"check_plan",return_value=None),patch.object(f.urllib.request,"urlopen",return_value=response) as network:
                result=f.get_window(plan,record["id"],target,allow_network=True)
                headers=network.call_args.args[0].headers
            self.assertIn("bytes=128-1023",list(headers.values()))
            self.assertEqual(result["original_window_sha256"],hashlib.sha256(data).hexdigest())

    def test_cannot_qualify_unmaterialized_full_61gb_tar(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):v.verify_all(SOURCE,t)

    def test_full_archive_lfs_digest_detects_corruption_without_real_data(self):
        with tempfile.TemporaryDirectory() as t:
            directory=Path(t)
            data=b"original-train"*45
            name="sheet"
            expected_hash=hashlib.sha256(data).hexdigest()
            fake={"bytes":len(data),"upstream_lfs_sha256":expected_hash}
            entry=dict(id="fake-first",source_domain=name,byte_start=0,
                byte_end_inclusive=len(data)-1,expected_length=len(data),
                upstream_full_tar_lfs_sha256=expected_hash)
            receipt=dict(schema="hala.h59.individual-train-tar-window-receipt.v1",
                original_window_id="fake-first",source_domain=name,
                original_commit_sha=m.REV,bytes_verified=len(data),
                range_start=0,range_end_inclusive=len(data)-1,
                original_window_sha256=expected_hash)
            (directory/"fake-first.tarwindow").write_bytes(data)
            (directory/"fake-first.receipt.json").write_text(json.dumps(receipt))
            with patch.object(v,"DOMAINS",{name:fake}):
                a=v.verify_archive(name,[entry],directory)
                self.assertEqual(a["original_tar_sha256"],expected_hash)
                self.assertFalse(a["task_records_parsed"])
                (directory/"fake-first.tarwindow").write_bytes(data[:-1]+b"Z")
                with self.assertRaises(ValueError):v.verify_archive(name,[entry],directory)

if __name__=="__main__":unittest.main()
