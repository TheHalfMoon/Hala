"""H410 trusted external source-review signatures, synthetic only.

Ephemeral test Ed25519 signing keys are memory-only and never an external trust
enrollment. Production trusted reviewer registry remains EMPTY.
"""
import base64,hashlib,importlib.util,time,unittest
from unittest.mock import patch

if importlib.util.find_spec("cryptography") is not None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
else:
    serialization=None
    Ed25519PrivateKey=None

from h201.original_native_action_executor import exact_original_loopback_url
from h410.signed_source_review import (
    SignedSourceReview,run_authenticated_reviewed_policy,canonical_payload,
    TRUSTED_SOURCE_REVIEWERS)
from h407.reviewed_action_bridge import action_digest
from test_h203_screenshot_bound_dispatch import Page,generated_960_png,one


class H410NoSiteFailClosedTests(unittest.TestCase):
    def test_no_trusted_reviewer_production_hard_stops_before_capture(self):
        self.assertEqual(dict(TRUSTED_SOURCE_REVIEWERS),{})
        page=Page(generated_960_png(14))
        called=[]
        with self.assertRaisesRegex(ValueError,"AUTHENTICATED_SOURCE_REVIEW_REQUIRED"):
            run_authenticated_reviewed_policy(
                page,"Click the unique visible button.",
                exact_original_loopback_url(8201),
                lambda *_:called.append(True),None)
        self.assertEqual(called,[])
        self.assertEqual(page.screenshots,0)
        self.assertEqual(page.events,[])


@unittest.skipUnless(Ed25519PrivateKey is not None,
                     "Ed25519 is optional on source-only no-site CI")
class H410SignedSourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.page=Page(generated_960_png(34))
        self.url=exact_original_loopback_url(8201)
        self.task="Click the unique red button."
        self.raw=one()
        self.private=Ed25519PrivateKey.generate()
        pub=self.private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.kid=hashlib.sha256(pub).hexdigest()
        self.registry={self.kid:{"identity":"Ephemeral owner fixture reviewer",
                                 "public_key_bytes":pub}}
        self.signs=0

    def reviewer(self,png,task,action,nonce,override=None):
        self.signs+=1
        now=int(time.time())
        data={
            "schema":"hala.h410.signed-visible-unique-target.v1",
            "source_review_purpose":"visually_verified_single_target",
            "nonce":nonce,"registered_url":self.url,
            "png_sha256":hashlib.sha256(png).hexdigest(),
            "task_sha256":hashlib.sha256(task.encode("utf8")).hexdigest(),
            "action_sha256":action_digest(action),
            "reviewer_identity":"Ephemeral owner fixture reviewer",
            "matching_targets":1,
            "visible_box":[180,150,220,190],
            "issued_unix":now,"expires_unix":now+15,
        }
        if override:data.update(override)
        sig=base64.b64encode(
            self.private.sign(canonical_payload(data))).decode("ascii")
        return SignedSourceReview(
            key_id=self.kid,payload=data,signature_b64=sig)

    def invoke(self,review=None,model=None):
        return run_authenticated_reviewed_policy(
            self.page,self.task,self.url,
            model or (lambda *_:self.raw),
            review or self.reviewer)

    def test_unenrolled_production_has_zero_keys_and_zero_model_calls(self):
        self.assertEqual(len(TRUSTED_SOURCE_REVIEWERS),0)
        called=[]
        with self.assertRaisesRegex(ValueError,"AUTHENTICATED_SOURCE_REVIEW_REQUIRED"):
            self.invoke(lambda *_: called.append(True))
        self.assertEqual(called,[])
        self.assertEqual(self.page.screenshots,0)
        self.assertEqual(self.page.events,[])

    def test_exact_signature_fresh_nonce_pixel_task_action_allows_test_click(self):
        with patch("h410.signed_source_review.TRUSTED_SOURCE_REVIEWERS",self.registry):
            result=self.invoke()
        self.assertEqual(self.page.events,[("click",200,170)])
        self.assertEqual(result["native_events"],1)
        self.assertTrue(result["signed_source_review_verified"])
        self.assertFalse(result["independent_ood_qualified"])
        self.assertFalse(result["real_gui_safety_certified"])
        self.assertEqual(self.signs,1)

    def test_nonce_replay_and_mismatched_signed_source_refused(self):
        invalid={
            "nonce":"f"*32,
            "png_sha256":"0"*64,
            "task_sha256":"0"*64,
            "action_sha256":"0"*64,
            "registered_url":"http://127.0.0.1:1/snap/index.html",
            "matching_targets":2,
            "visible_box":[0,0,90,90],
            "reviewer_identity":"Unregistered attacker signer",
            "expires_unix":1,
        }
        for field,value in invalid.items():
            with self.subTest(field=field):
                self.page.events=[]
                with patch("h410.signed_source_review.TRUSTED_SOURCE_REVIEWERS",self.registry):
                    with self.assertRaisesRegex(
                         ValueError,"AUTHENTICATED_SOURCE_REVIEW_REQUIRED"):
                        self.invoke(lambda p,t,a,n:
                            self.reviewer(p,t,a,n,{field:value}))
                self.assertEqual(self.page.events,[])

    def test_unknown_signer_refused_even_with_valid_signature(self):
        with patch("h410.signed_source_review.TRUSTED_SOURCE_REVIEWERS",{
                    "0"*64:{"identity":"Other verified fixture reviewer",
                            "public_key_bytes":b"1"*32}}):
            with self.assertRaisesRegex(ValueError,"AUTHENTICATED_SOURCE_REVIEW_REQUIRED"):
                self.invoke()
        self.assertEqual(self.page.events,[])

    def test_corrupt_signature_refused_with_no_native_input(self):
        def corrupted(p,t,a,n):
            doc=self.reviewer(p,t,a,n)
            return SignedSourceReview(
                key_id=doc.key_id,payload=doc.payload,
                signature_b64="A"*88)
        with patch("h410.signed_source_review.TRUSTED_SOURCE_REVIEWERS",self.registry):
            with self.assertRaisesRegex(ValueError,"AUTHENTICATED_SOURCE_REVIEW_REQUIRED"):
                self.invoke(corrupted)
        self.assertEqual(self.page.events,[])

    def test_review_failure_redacted_and_no_browser_event(self):
        def bad(_p,_t,_a,_n):
            raise RuntimeError("UNTRUSTED_PRIVATE_PATIENT_SECRET_NOT_TO_LOG")
        with patch("h410.signed_source_review.TRUSTED_SOURCE_REVIEWERS",self.registry):
            with self.assertRaises(ValueError) as caught:
                self.invoke(bad)
        self.assertNotIn("PATIENT_SECRET",str(caught.exception))
        self.assertEqual(self.page.events,[])

    def test_signed_approval_cannot_authorize_stale_page(self):
        def altered(p,t,a,n):
            doc=self.reviewer(p,t,a,n)
            self.page.image=generated_960_png(35)
            return doc
        with patch("h410.signed_source_review.TRUSTED_SOURCE_REVIEWERS",self.registry):
            with self.assertRaisesRegex(ValueError,"stale"):
                self.invoke(altered)
        self.assertEqual(self.page.events,[])

    def test_abstain_without_signature_does_not_send_native_event(self):
        self.raw=one("abstain")
        with patch("h410.signed_source_review.TRUSTED_SOURCE_REVIEWERS",self.registry):
            result=self.invoke(lambda *_:self.fail("abstain must not summon signer"))
        self.assertEqual(result["native_events"],0)
        self.assertEqual(self.page.events,[])
        self.assertFalse(result["signed_source_review_verified"])

if __name__=="__main__":unittest.main()
