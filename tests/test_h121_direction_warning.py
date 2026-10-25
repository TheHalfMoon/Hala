"""Post-hoc H121 direction warning is descriptive, never model admission."""
import unittest
from h121.geometry import (training_vs_prediction_diagnostic,
                           summarize_model_predictions,SCHEMA)

class WarningTests(unittest.TestCase):
    def setUp(self):
        self.train={"source_split":"TRAIN_ONLY","unique_source_images":512,
          "true_train_displacement":{"count":512,
             "most_common_vector_fraction":.0117,
             "near_zero_fraction":.2246,"positive_x_fraction":.502}}
        predicted={"count":64,"most_common_vector_fraction":.7344,
                   "near_zero_fraction":.0312,"positive_x_fraction":.0312}
        self.model={"prediction_only":True,
            "no_true_EVAL_locations_or_boxes_used":True,
            "predicted_signed_displacement":predicted}
    def test_repeated_direction_model_displays_warning_not_quality_verdict(self):
        r=training_vs_prediction_diagnostic(self.train,self.model)
        self.assertTrue(r["repeat_single_displacement_warning"])
        self.assertTrue(r["rightward_direction_marginal_drift_warning"])
        self.assertTrue(r["any_direction_or_magnitude_collapse_warning"])
        self.assertTrue(r["posthoc_descriptive_warning_not_model_selection"])
        self.assertTrue(r["original_H61_success_required_for_GUI_qualification"])
        self.assertTrue(r["distribution_shift_is_not_causal_proof"])
        self.assertNotIn("model_qualified",r)
    def test_rejects_unverified_train_or_model_summary(self):
        for bad in ({**self.train,"source_split":"eval"},
                    {**self.train,"unique_source_images":511}):
            with self.assertRaises(ValueError):
                training_vs_prediction_diagnostic(bad,self.model)
        with self.assertRaises(ValueError):
            training_vs_prediction_diagnostic(self.train,{
                **self.model,"no_true_EVAL_locations_or_boxes_used":False})
        with self.assertRaises(ValueError):
            training_vs_prediction_diagnostic(self.train,{
                **self.model,"predicted_signed_displacement":{
                    **self.model["predicted_signed_displacement"],"count":16}})
    def test_normal_predicted_direction_marginals_not_flagged(self):
        model={**self.model,"predicted_signed_displacement":{
          **self.model["predicted_signed_displacement"],
          "most_common_vector_fraction":.02,
          "near_zero_fraction":.2,"positive_x_fraction":.48}}
        r=training_vs_prediction_diagnostic(self.train,model)
        self.assertFalse(r["any_direction_or_magnitude_collapse_warning"])

if __name__=="__main__":unittest.main()
