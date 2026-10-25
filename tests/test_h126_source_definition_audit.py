"""H126 source-only instruction/geometry audit, never claims semantic proof."""
import unittest
from h126.source_audit import (Source,propose,metadata_summary,TOTAL,
                              PER_FAMILY,PREVIOUS)
from h123.schedule import FAMILIES

def fixture():
    cases=[]
    for family in FAMILIES:
        for n in range(PER_FAMILY):
            cases.append({
                "id":f"synthetic-{family}-{n}",
                "family":family,"split":"train",
                "image_width":1000,"image_height":600,
                "intent":("Move the selected slider to the right "
                          if n%2 else "Select this GUI item"),
                "start_bbox":[100,150,108,158],
                "end_bbox":[300,450,306,456]})
    return Source(tuple(cases),tuple(range(TOTAL)),{
        "original_source_scenes":64,"schedule_sha256":"a"*64,
        "source_cases_SHA":"b"*64})

class H126AuditTests(unittest.TestCase):
    def test_all_prior_source_images_excluded_by_contract(self):
        self.assertEqual(PREVIOUS,2624)
        self.assertEqual(TOTAL,64)
        self.assertEqual(TOTAL,len(FAMILIES)*PER_FAMILY)
        with self.assertRaises(ValueError):propose([],None)

    def test_aggregate_does_not_expose_raw_source_instructions_or_boxes(self):
        result=metadata_summary(fixture())
        self.assertEqual(result["original_untouched_screenshots"],TOTAL)
        self.assertFalse(result["semantic_identifiability_inferred"])
        self.assertEqual(len(result["per_domain_aggregates"]),4)
        self.assertEqual(result["excluded_all_prior_2624_source_images"],2624)
        serialized=str(result)
        self.assertNotIn("Move the selected slider",serialized)
        self.assertNotIn("'start_bbox'",serialized)
        for agg in result["per_domain_aggregates"].values():
            self.assertEqual(agg["original_source_images"],16)
            self.assertEqual(agg["median_start_box_width_px"],9)
            self.assertEqual(agg["median_end_box_width_px"],7)

    def test_bad_source_geometry_refused(self):
        source=fixture()
        bad=list(source.cases)
        bad[0]={**bad[0],"end_bbox":[-1,4,6,8]}
        with self.assertRaises(ValueError):
            metadata_summary(Source(tuple(bad),source.indices,source.summary))
