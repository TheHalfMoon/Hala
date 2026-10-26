"""Static contracts for BLIND original-source capture: never train on it."""
import unittest
from h140.blind_original_capture import board_for,param,moved,SCHEMA
from h140.blind_source_preflight import SOURCES


class H140SealedPrivateBlindSourceContracts(unittest.TestCase):
    def test_case_schedule_bilingual_and_native_layout_diverse(self):
        p=[param(i+30000) for i in range(72)]
        self.assertEqual(sum(x[3]=="en" for x in p),36)
        self.assertEqual(sum(x[3]=="ar" for x in p),36)
        self.assertGreaterEqual(len({(x[0],x[1],x[2]) for x in p}),20)
        self.assertIn("private",SCHEMA)
        for n,src,end,lang,scroll in p:
            self.assertTrue(4<=n<=9 and 0<=src<n and 0<=end<n)

    def test_exact_original_app_card_status_transition_not_oracle_fallback(self):
        for app in SOURCES:
            obj=board_for(app,30000,5,2)
            self.assertTrue(obj["id"].startswith("h140-private-blind-qa-task"))
            self.assertIn(obj["storage"],("kanbanTasks","kanbanBoards"))
            if app=="ood-vanilla-complete":
                state=obj["seed"]
                self.assertFalse(moved(app,obj,state))
                record=next(t for t in state if t["id"]==obj["id"])
                record["status"]=obj["destination"]
            else:
                state=obj["seed"]
                self.assertFalse(moved(app,obj,state))
                record=next(t for t in state[0]["tasks"] if t["id"]==obj["id"])
                record["status"]=obj["destination"]
            self.assertTrue(moved(app,obj,state))
            record["status"]=obj["source"]
            self.assertFalse(moved(app,obj,state))


if __name__=="__main__":
    unittest.main()
