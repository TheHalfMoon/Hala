"""H407 source-confirmed action boundary; all synthetic PNGs, no GPU."""
import unittest
from hashlib import sha256
from h407.reviewed_action_bridge import (
    VisualTargetReview, run_source_reviewed_policy, action_digest)
from h201.original_native_action_executor import exact_original_loopback_url
from test_h203_screenshot_bound_dispatch import Page,generated_960_png,one


class H407SourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.png=generated_960_png(78)
        self.page=Page(self.png)
        self.url=exact_original_loopback_url(8201)
        self.task="Click the unique visible square."
        self.raw=one()
        self.reviews=0

    def proof(self,png,task,action,*,matches=1,box=(180,150,220,190)):
        self.reviews+=1
        return VisualTargetReview(
            png_sha256=sha256(png).hexdigest(),
            task_sha256=sha256(task.encode("utf8")).hexdigest(),
            action_sha256=action_digest(action),
            matching_targets=matches,
            visible_box=box,
        )

    def invoke(self,review=None,model=None):
        if model is None:
            model=lambda _png,_prompt:self.raw
        return run_source_reviewed_policy(self.page,self.task,self.url,
                                          model,review)

    def test_click_requires_own_separate_review_before_native_input(self):
        result=self.invoke(self.proof)
        self.assertEqual(self.page.events,[("click",200,170)])
        self.assertEqual(self.reviews,1)
        self.assertEqual(self.page.screenshots,2)
        self.assertTrue(result["local_review_bound_to_png"])
        self.assertFalse(result["independent_ood_qualified"])
        self.assertFalse(result["real_gui_safety_certified"])

    def test_missing_reviewer_blocks_before_capture_or_predict(self):
        calls=[]
        with self.assertRaisesRegex(ValueError,"review"):
            self.invoke(None,lambda *_:calls.append(True))
        self.assertEqual(self.page.screenshots,0)
        self.assertEqual(calls,[])
        self.assertEqual(self.page.events,[])

    def test_zero_or_two_targets_never_dispatch(self):
        for count in (0,2,3,-1,True):
            with self.subTest(count=count):
                self.page.events=[]
                with self.assertRaisesRegex(ValueError,"unique"):
                    self.invoke(lambda p,t,a:self.proof(p,t,a,matches=count))
                self.assertEqual(self.page.events,[])

    def test_box_not_containing_predicted_click_refuses(self):
        with self.assertRaisesRegex(ValueError,"visible target"):
            self.invoke(lambda p,t,a:self.proof(p,t,a,box=(0,0,50,50)))
        self.assertEqual(self.page.events,[])

    def test_wrong_frame_or_task_or_action_digest_refuses(self):
        for field in ("png_sha256","task_sha256","action_sha256"):
            with self.subTest(field=field):
                def review(p,t,a):
                    proof=self.proof(p,t,a)
                    values={k:getattr(proof,k) for k in (
                        "png_sha256","task_sha256","action_sha256",
                        "matching_targets","visible_box")}
                    values[field]="0"*64
                    return VisualTargetReview(**values)
                with self.assertRaisesRegex(ValueError,"bind"):
                    self.invoke(review)
                self.assertEqual(self.page.events,[])

    def test_stale_frame_during_review_never_dispatches(self):
        def change(p,t,a):
            self.page.image=generated_960_png(79)
            return self.proof(p,t,a)
        with self.assertRaisesRegex(ValueError,"stale"):
            self.invoke(change)
        self.assertEqual(self.page.events,[])

    def test_navigated_page_during_review_never_dispatches(self):
        def change(p,t,a):
            self.page.url="https://untrusted.example/"
            return self.proof(p,t,a)
        with self.assertRaisesRegex(ValueError,"browser left"):
            self.invoke(change)
        self.assertEqual(self.page.events,[])

    def test_drag_and_keyboard_require_other_explicit_review_contract(self):
        for raw in (one("drag",start=[100,100],end=[110,110]),
                    one("key_press",point=[200,170],key="Enter"),
                    one("scroll",point=[200,170],dy=10)):
            self.raw=raw
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError,"click-only"):
                    self.invoke(self.proof)
                self.assertEqual(self.page.events,[])

    def test_abstain_never_dispatches_and_never_invokes_review(self):
        self.raw=one("abstain")
        result=self.invoke(lambda *_:self.fail("review must not run for abstention"))
        self.assertEqual(result["native_events"],0)
        self.assertEqual(self.page.events,[])
        self.assertFalse(result["local_review_bound_to_png"])

    def test_bad_reviewer_type_or_exception_fails_closed(self):
        for bad in (None,True,{"matching_targets":1},"approved"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.invoke(lambda *_:bad)
                self.assertEqual(self.page.events,[])
        with self.assertRaisesRegex(ValueError,"review declined"):
            self.invoke(lambda *_: (_ for _ in ()).throw(RuntimeError("SECRET_REVIEWER_NOT_DISCLOSED")))
        self.assertEqual(self.page.events,[])

    def test_invalid_source_box_coords_fail_closed(self):
        for box in ((-1,0,30,30),(1,1,1,6),(1,1,960,20),
                    (True,1,50,50),(1,1,float("nan"),50)):
            with self.subTest(box=box):
                with self.assertRaisesRegex(ValueError,"visible target"):
                    self.invoke(lambda p,t,a:self.proof(p,t,a,box=box))
                self.assertEqual(self.page.events,[])

if __name__=="__main__":
    unittest.main()
