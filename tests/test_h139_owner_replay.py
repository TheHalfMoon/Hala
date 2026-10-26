"""Independent original-app replay accepts only pinned raw neural point results."""
import unittest
from h139.owner_replay import ARMS,NEURAL_SCHEMA,prediction_pairs


class H139OwnerReplayGuards(unittest.TestCase):
    def test_32_exact_raw_native_calls_and_source_integrity(self):
        owner="a"*64
        adapter="b"*64
        rows=[{"id":f"blind-{i}","image_sha256":f"{i+1:064x}"} for i in range(8)]
        source={"cases":rows}
        report={"schema":NEURAL_SCHEMA,
                "status":"H139_COMPLETE_32_REAL_BLIND_NEURAL_CALLS",
                "original_owner_manifest_sha256":owner,"model_calls":32,
                "optimizer_steps":0,"source_role":"blind_test",
                "paid_compute_used":False,"main_35B_updated":False,
                "adapter_sha256":adapter}
        items=[{"arm":arm,"id":row["id"],"image_sha256":row["image_sha256"],
                "role":role,"raw":"no model point","point":None}
               for arm in ARMS for row in rows for role in ("start","end")]
        points=prediction_pairs(source,report,items,owner,adapter)
        self.assertEqual(len(points),32)
        self.assertTrue(all(point is None for point in points.values()))
        with self.assertRaises(ValueError):
            prediction_pairs(source,dict(report,adapter_sha256="c"*64),items,owner,adapter)
        tampered=[dict(x) for x in items]
        tampered[0]["point"]=[480,360]
        with self.assertRaises(ValueError):
            prediction_pairs(source,report,tampered,owner,adapter)
        with self.assertRaises(ValueError):
            prediction_pairs(source,report,items[:-1],owner,adapter)


if __name__=="__main__":
    unittest.main()
