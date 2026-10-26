"""H130 screenshot-only policy boundary and feedback-loop contracts."""
import unittest
from h130.browser_env import Observation, Feedback
from h130.rollout import run_episode

class Fake:
    def __init__(self, task):
        self.task=task
        self.i=0
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def observe(self):
        return Observation("Drag blue to green",("scene"+str(self.i)).encode(),960,720,self.i)
    def execute_drag(self,start,end):
        self.i+=1
        success=self.i==2
        return Feedback(success,self.observe(),self.i,True,success or self.i>=3)

class H130PolicyTests(unittest.TestCase):
    def test_screen_only_policy_receives_after_action_feedback(self):
        seen=[]
        def policy(obs,feedback):
            self.assertEqual(set(obs.__dict__),
                {"instruction","screenshot_png","width","height","step"})
            seen.append((obs.step, None if feedback is None else feedback.success))
            return ((1,2),(3,4))
        r=run_episode("slider",policy,factory=Fake)
        self.assertTrue(r.success)
        self.assertEqual(r.policy_calls,2)
        self.assertEqual(seen,[(0,None),(1,False)])
        self.assertNotEqual(r.attempts[0].before_image_sha256,
                            r.attempts[0].after_image_sha256)
        self.assertFalse(r.trained_model_inference_verified)

    def test_bounded_rejects_bad_policy_outputs(self):
        with self.assertRaises(ValueError):
            run_episode("not-a-task",lambda o,f:((0,0),(1,1)),factory=Fake)
        with self.assertRaises(ValueError):
            run_episode("drop",lambda o,f:None,factory=Fake)
        with self.assertRaises(ValueError):
            run_episode("drop",lambda o,f:((1,2),(3,4)),factory=Fake,max_actions=8)

    def test_policy_not_claimed_neural(self):
        rec=run_episode("drop",lambda o,f:((0,0),(1,1)),
                        factory=Fake,max_actions=1)
        self.assertFalse(rec.success)
        self.assertEqual(rec.policy_calls,1)
        self.assertFalse(rec.trained_model_inference_verified)

if __name__=="__main__":
    unittest.main()
