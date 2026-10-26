import unittest
from h135.pilot_decision import choose

class H135FrozenPilotSelectionTests(unittest.TestCase):
    def fixture(self):
        m=lambda s,h,e:{"n":12,"real_gui_successes":s,
             "start_hits":h,"median_start_error_px":e}
        return {"status":"H135_PILOT_PRIVATE_96_NEURAL_AND_REAL_BROWSER_SCORED",
                "real_neural_calls":96,"untouched_H135_sealed_holdout":True,
                "models":{"h134_adapter__downsample_640":m(2,4,60),
                          "h134_adapter__native_960":m(3,3,5)}}

    def test_full_task_primary_over_secondary(self):
        data=self.fixture()
        self.assertEqual(choose(data)["selected_mode"],"native_960")
        data["models"]["h134_adapter__native_960"]["real_gui_successes"]=2
        self.assertEqual(choose(data)["selected_mode"],"downsample_640")
        data["models"]["h134_adapter__native_960"]["start_hits"]=4
        self.assertEqual(choose(data)["selected_mode"],"native_960")

    def test_invalid_source_refuses_and_native_exact_tie(self):
        data=self.fixture()
        data["models"]["h134_adapter__native_960"]=dict(data["models"]["h134_adapter__downsample_640"])
        self.assertEqual(choose(data)["selected_mode"],"native_960")
        data["untouched_H135_sealed_holdout"]=False
        with self.assertRaises(ValueError):choose(data)
if __name__=="__main__":unittest.main()
