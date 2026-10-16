"""Stdlib negative guards using saved DEV records; never fake a TEST success."""
import ast, copy, hashlib, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import h22_test_restart_audit as audit

ROOT=Path(__file__).resolve().parents[1]
OLD_UUID='GPU-24febf58-cdfb-1208-90d1-a9b3b95c03c1'
DIFFERENT_UUID='GPU-11111111-2222-3333-4444-555555555555'
class RestartAuditGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record=json.loads((ROOT/'outputs/h22-attempt2-profile-0-fp16-dev-evidence/run/result.json').read_bytes())
        cls.amendment=json.loads((ROOT/'outputs/h22-test-restart-prospective-amendment.json').read_bytes())
        # Synthetic manifest UUID solely for testing validation. No qualification
        # or actual TEST result is asserted by this configuration fixture.
        cls.amendment['gpu_uuid']=OLD_UUID
    def test_local_three_indices_map_only_to_original_test_indices(self):
        for local,mode in enumerate(('fp16','int8','nf4')):
            expected=audit.expected_identity(self.amendment,OLD_UUID,local)
            self.assertEqual((expected['mode'],expected['split'],expected['logical_original_index']),(mode,'test',local+3))
        for bad in (-1,3,True):
            with self.assertRaises(ValueError): audit.expected_identity(self.amendment,OLD_UUID,bad)
    def test_pending_or_wrong_external_uuid_rejected(self):
        for uuid in ('PENDING_ACTUAL_QUALIFICATION',DIFFERENT_UUID):
            with self.assertRaises(ValueError): audit.expected_identity(self.amendment,uuid,0)
    def test_dev_manifest_or_group_cannot_be_substituted(self):
        for key,value in [('schema','hala.h22.prospective_experiment.v1'),('attempt',2),('mode_order',[['fp16','dev']]),('dev_experiment_sha256','f'*64),('dev_retention',[]),('source_commit','f'*40),('source_freeze_sha256','f'*64),('adapter_manifest_sha256','f'*64),('adoption_qualified',True),('quality_qualified',True)]:
            amended=copy.deepcopy(self.amendment); amended[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): audit.expected_identity(amended,OLD_UUID,0)
    def test_actual_saved_dev_cannot_pass_as_restart_test_even_when_uuid_mode_match(self):
        before=copy.deepcopy(self.record)
        errors=audit.validate_record_identity(self.record,audit.expected_identity(self.amendment,OLD_UUID,0))
        self.assertIn('actual record split differs from fixed TEST group',errors)
        self.assertEqual(self.record,before)
    def test_actual_saved_dev_wrong_expected_uuid_is_separately_rejected(self):
        expected=audit.expected_identity(self.amendment,OLD_UUID,0)|{'gpu_uuid':DIFFERENT_UUID}
        self.assertIn('actual record UUID differs from explicit expected UUID',audit.validate_record_identity(self.record,expected))
    def test_actual_saved_dev_wrong_expected_mode_is_separately_rejected(self):
        expected=audit.expected_identity(self.amendment,OLD_UUID,1)
        self.assertIn('actual record mode differs from fixed local index',audit.validate_record_identity(self.record,expected))
    def test_public_amendment_hash_required_before_record_access(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'amendment.json'; path.write_text(json.dumps(self.amendment),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'External amendment SHA mismatch'):
                audit.audit_restart(SimpleNamespace(amendment=path,amendment_sha256='f'*64))
    def test_immutable_old_auditor_and_new_syntax(self):
        old=(ROOT/'work/h22_actual_precision_audit.py').read_bytes()
        self.assertEqual(hashlib.sha256(old).hexdigest(),'a816cde40e58b361f7143e5adb165006a877b5a9c39ef9b88540a59e05d1a09f')
        ast.parse((ROOT/'work/h22_test_restart_audit.py').read_text(encoding='utf-8'))
if __name__=='__main__': unittest.main()
