"""H111 frozen H110 method + new source EVAL 48..55 independent replication."""
import unittest
from h111.eval import START,COUNT,END,ARMS,ZOOM_SIZE,frozen,score
from h110.eval import ARMS as H110_ARMS,ZOOM_SIZE as H110_ZOOM,score as H110_SCORE

class TestH111PreregisteredReplication(unittest.TestCase):
    def test_32_new_source_scenes_never_seen_before(self):
        self.assertEqual((START,COUNT,END),(48,8,56))
        source=[{"id":f"unseen-original-eval-{i}"} for i in range(56)]
        result=frozen(source)
        self.assertEqual(len(result),8)
        self.assertEqual(result[0]["id"],"unseen-original-eval-48")
        self.assertEqual(result[-1]["id"],"unseen-original-eval-55")
        with self.assertRaises(ValueError):frozen(source[:-1])
        original=list(source)
        original[52]=original[50]
        with self.assertRaises(ValueError):frozen(original)
    def test_same_treatment_original_scorer_and_compute_arms(self):
        self.assertEqual(ARMS,H110_ARMS)
        self.assertEqual(ZOOM_SIZE,H110_ZOOM)
        self.assertEqual(ZOOM_SIZE,512)
        self.assertIs(score,H110_SCORE)
if __name__=="__main__":unittest.main()
