"""H59 MiniMax source/handoff consistency: CPU, no external accounts, no weights."""
import hashlib,json,unittest
from pathlib import Path
from h59.master_data_plan import check_plan

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"h59"
class H59MiniMaxHandoffTests(unittest.TestCase):
    def test_source_catalog_and_signed_handoff_claim_scope(self):
        data=json.loads((FOLDER/"full_source_windows.json").read_text("utf8"))
        summary=check_plan(data)
        receipt=json.loads((FOLDER/"HANDOFF_RECEIPT.json").read_text("utf8"))
        source=hashlib.sha256((FOLDER/"full_source_windows.json").read_bytes()).hexdigest()
        self.assertEqual(receipt["private_kaggle_map_manifest_sha256"],source)
        self.assertEqual(summary["planned_bytes"],receipt["original_train_tar_total_bytes"])
        self.assertEqual(summary["planned_windows"],receipt["planned_download_windows"])
        self.assertEqual(summary["upstream_tasks"],3560142)
        self.assertEqual(receipt["downloaded_new_full_source_windows"],0)
        self.assertEqual(receipt["fully_materialized_new_train_labels"],0)
        self.assertEqual(receipt["preexisting_verified_training_actions"],2346)
        self.assertFalse(receipt["actual_checkpoint_trained"])
        self.assertEqual(receipt["real_gpu_optimizer_updates"],0)

    def test_private_map_verified_but_no_fake_three_million_in_kaggle(self):
        r=json.loads((FOLDER/"HANDOFF_RECEIPT.json").read_text("utf8"))
        self.assertTrue(r["private_kaggle_map_authenticated_readback_sha_verified"])
        self.assertTrue(r["private_kaggle_map_unauthenticated_http_403"])
        self.assertTrue(r["private_kaggle_map_provider_private_flag_listed"])
        self.assertTrue(r["private_kaggle_map_is_metadata_only"])
        self.assertFalse(r["minimax_training_job_started"])
        self.assertFalse(r["heldout_model_neural_results_attested"])

    def test_h59_progress_in_repository_catalog_remains_source_only(self):
        doc=json.loads((ROOT/"research/data_library/catalog.json").read_text("utf8"))
        self.assertEqual(len(doc["datasets"]),14)
        rows=[x for x in doc["source_maps"] if x["id"]=="hala-h59-millions-source-map"]
        self.assertEqual(len(rows),1)
        row=rows[0]
        self.assertEqual(row["planned_original_tar_windows"],116)
        self.assertEqual(row["actual_new_full_source_windows_downloaded"],0)
        self.assertFalse(row["actual_3560142_train_tasks_private_kaggle_stored"])
        self.assertFalse(row["training_completed"])
        self.assertEqual(row["actual_gpu_steps"],0)
        self.assertTrue(all(not x["progress"]["training_completed"] for x in doc["datasets"]))

    def test_map_prompt_runnable_and_prior_data_source_separate(self):
        prompt=(FOLDER/"MINIMAX_EXECUTION_PROMPT.md").read_text("utf8")
        roadmap=(FOLDER/"EXECUTION_MAP.md").read_text("utf8")
        progress=(FOLDER/"MINIMAX_PROGRESS.md").read_text("utf8")
        for name in ("h59/full_source_windows.json","h59.fetch_window",
                     "3,560,142","286,012","ordered=false","12-hour",
                     "PRIVATE","DONE"):
            self.assertTrue(name in prompt+roadmap+progress,name)
        self.assertIn("commercial",prompt.lower())
        self.assertIn("Vercel",prompt)
        self.assertIn("benchmark",roadmap.lower())
        self.assertIn("NOT DONE",progress)
        self.assertIn("204",progress)
        self.assertIn("2,346",progress)
        self.assertNotIn("3,560,142 private stored records",progress)

    def test_all_private_source_data_excluded_from_new_source_plan(self):
        for path in FOLDER.iterdir():
            if path.is_file():
                self.assertNotIn(path.suffix.lower(),(".jpg",".jpeg",".png",".tar",".zip",".safetensors",".pt"))
        x=json.loads((FOLDER/"full_source_windows.json").read_text("utf8"))
        self.assertTrue(all(item["source_window_sha256"] is None for item in x["windows"]))
if __name__=="__main__":unittest.main()
