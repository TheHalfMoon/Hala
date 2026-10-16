"""Stdlib-only H23 operator contract checks. No GPU, downloads or subprocess execution."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import copy
import signal

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('h23_operator_contract',HERE/'h23-operator.py')
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)


class OperatorBasics(unittest.TestCase):
    barrier_original = staticmethod(operator.barrier)
    @classmethod
    def setUpClass(cls):
        cls.helper = operator.verified_h22_operator(HERE/'h22-test-only-operator.py')
        cls.original = json.loads((HERE/'h22-prospective-experiment.json').read_text())

    def test_invalid_amendment_stops_before_allocation_and_retains_raw_failure(self):
        self.drive(bad_amendment=True)

    def test_guarded_run_checks_six_public_exports_before_mocked_allocation(self):
        self.drive()

    def drive(self,bad_amendment=False):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            table,test,test_sha,fetched = self.fixtures(root/'h22')
            amendment,_ = self.amendments()
            if bad_amendment: amendment['source_commit']='f'*40
            amendment_raw = json.dumps(amendment).encode()
            table_raw = json.dumps(table).encode()
            metadata = {
                operator.PREFIX+operator.PROSPECT_COMMIT+'/h23-prospective-20261016/experiment.json':(HERE/'h23-prospective-experiment.json').read_bytes(),
                operator.PREFIX+operator.SOURCE+'/experiments/serving-lifecycle/source-freeze.json':(HERE.parent/'work/Hala-lifecycle/experiments/serving-lifecycle/source-freeze.json').read_bytes(),
                operator.PREFIX+'c'*40+'/'+operator.AMENDMENT_PATH:amendment_raw,
                operator.PREFIX+'d'*40+'/'+operator.TABLE_PATH:table_raw,
                operator.PREFIX+amendment['test_manifest_commit']+'/h22-test-restart/experiment.json':json.dumps(test).encode(),
                self.helper.ORIGINAL_URL:(HERE/'h22-prospective-experiment.json').read_bytes()}
            def metadata_fetch(url,expected,limit):
                raw = metadata[url]
                if len(raw)>limit or operator.sha(raw)!=expected: raise ValueError('CPU metadata SHA mismatch')
                return raw
            def prepared():
                self.assertEqual(public.call_count,12)
                return root/'source'
            def invoked(source,evidence,report):
                report['status']='incomplete_timeout'
                return report
            with patch.object(operator,'EXECUTION_READY',True),patch.object(operator,'genuine_provider'), \
                    patch.object(operator,'EVIDENCE',root/'evidence'),patch.object(operator,'ROOT',root/'h23'), \
                    patch.object(operator,'verified_h22_operator',return_value=self.helper), \
                    patch.object(operator,'fetch',side_effect=metadata_fetch), \
                    patch.object(self.helper,'fetch',side_effect=fetched) as public, \
                    patch.object(operator,'prepare_source_and_cache',side_effect=prepared) as prepare, \
                    patch.object(operator,'qualify',return_value={'gpu_uuid':test['gpu_uuid']}), \
                    patch.object(operator,'clean_source'),patch.object(operator,'invoke',side_effect=invoked) as invoke, \
                    patch.object(operator,'barrier',wraps=lambda raw,sha,helper,test,group_sha: \
                        OperatorBasics.barrier_original(raw,sha,helper,test,group_sha,root/'h22')), \
                    patch('builtins.print'):
                report = operator.run('c'*40,operator.sha(amendment_raw),'d'*40,operator.sha(table_raw))
            self.assertFalse(report['actual_qualification'])
            self.assertEqual(prepare.call_count,0 if bad_amendment else 1)
            self.assertEqual(invoke.call_count,0 if bad_amendment else 1)
            with zipfile.ZipFile(root/'evidence.zip') as retained:
                self.assertIn('hardware-retention-amendment.json',retained.namelist())
                self.assertIn('source-freeze.json',retained.namelist())
                self.assertIn('invocation.log',retained.namelist())

    def test_reused_helper_hash_is_checked_before_import(self):
        with tempfile.TemporaryDirectory() as root:
            helper = Path(root)/'helper.py'
            helper.write_bytes(b'raise AssertionError("must not import changed bytes")')
            with patch.object(operator.importlib.util,'spec_from_file_location') as imported:
                with self.assertRaises(ValueError): operator.verified_h22_operator(helper)
                imported.assert_not_called()

    def test_immutable_fetch_bounds_digest_and_redirects(self):
        url = operator.PREFIX+'a'*40+'/h23-prospective-20261016/experiment.json'
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def geturl(self): return url
            def read(self,bound): return b'x'*bound
        with patch.object(operator.urllib.request,'urlopen',return_value=Response()) as network:
            with self.assertRaises(ValueError): operator.fetch(url,'b'*64,10)
            with self.assertRaises(ValueError): operator.fetch(operator.PREFIX+'main/file','b'*64,10)
            self.assertEqual(network.call_count,1)
        class Redirect(Response):
            def geturl(self): return 'https://elsewhere.invalid/file'
        with patch.object(operator.urllib.request,'urlopen',return_value=Redirect()):
            with self.assertRaises(ValueError): operator.fetch(url,'b'*64,10)

    def test_exact_h23_and_test_amendment_source_hardware_groups(self):
        amendment,test = self.amendments()
        operator.validate_amendment(amendment,test,self.helper,self.original)
        for key,value in [('source_commit','f'*40),('source_freeze_sha256','f'*64),
                          ('scientific_changes',True),('cross_gpu_cost_pairing',True),
                          ('h23_expected_gpu_uuid','GPU-11111111-2222-3333-4444-555555555555'),
                          ('test_manifest_commit',None)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                operator.validate_amendment(dict(amendment,**{key:value}),test,self.helper,self.original)
        for key,value in [('attempt',2),('source_commit','f'*40),('dev_experiment_sha256','f'*64),
                          ('mode_order',[['fp16','dev']]),('dev_retention',[])]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                operator.validate_amendment(amendment,dict(test,**{key:value}),self.helper,self.original)

    def amendments(self):
        test = json.loads((HERE/'h22-test-restart-prospective-amendment.json').read_text())
        test['gpu_uuid'] = 'GPU-99999999-2222-3333-4444-555555555555'
        raw = json.dumps(test).encode()
        amendment = json.loads((HERE/'h23-hardware-retention-amendment.json').read_text())
        amendment.update(test_manifest_commit='a'*40,test_manifest_sha256=operator.sha(raw),h23_expected_gpu_uuid=test['gpu_uuid'])
        return amendment,test

    def fixtures(self,root,*,bad_test_attempt=False,forged_result=False):
        amendment,test = self.amendments()
        experiment = json.dumps(test).encode()
        test_sha = operator.sha(experiment)
        rows,remote = [],{}
        for item in self.helper.DEV_RETENTION:
            index = item['index']
            name = 'profile-'+str(index)+'-'+('fp16','int8','nf4')[index]+'-dev'
            prefix = operator.PREFIX+item['commit']+'/h22-attempt2/'+name
            remote[prefix+'-evidence.zip'] = (HERE/('h22-attempt2-'+name+'-evidence.zip')).read_bytes()
            remote[prefix+'-receipt.json'] = (HERE.parent/'work/Hala-artifacts/h22-attempt2'/(name+'-receipt.json')).read_bytes()
            rows.append(dict(item,group='dev'))
        for index,mode in enumerate(('fp16','int8','nf4')):
            name = 'profile-'+str(index)+'-'+mode+'-test'
            status = {'schema':'hala.h22.single_profile.v1','index':index,'name':name,
                      'source_commit':self.helper.COMMIT,'experiment_sha256':test_sha,
                      'status':'failed','returncode':1,'argv':['python','--mode',mode,'--split','test']}
            if forged_result: status['result_sha256']='f'*64
            stream = io.BytesIO()
            with zipfile.ZipFile(stream,'w') as archive:
                archive.writestr('experiment.json',experiment)
                archive.writestr('profile-status.json',json.dumps(status))
                archive.writestr('profile.log','CPU fixture failure, no model execution')
                archive.writestr('operator-launcher.py',(HERE/'h22-test-only-operator.py').read_bytes())
                archive.writestr('qualification.json',json.dumps({'attempt':2 if bad_test_attempt else 3,
                    'source_commit':self.helper.COMMIT,'source_freeze_sha256':self.helper.FREEZE,'gpu_uuid':test['gpu_uuid']}))
            raw = stream.getvalue()
            archive_sha = operator.sha(raw)
            receipt = {'schema':'hala.h22.public_retention_receipt.v1','index':index,
                       'experiment_sha256':test_sha,'archive_sha256':archive_sha,
                       'archive_path':'h22-test-restart/'+name+'-evidence.zip'}
            receipt_raw = json.dumps(receipt).encode()
            commit = 'b'*40
            prefix = operator.PREFIX+commit+'/h22-test-restart/'+name
            remote[prefix+'-evidence.zip'],remote[prefix+'-receipt.json'] = raw,receipt_raw
            folder = root/'precision-test-restart'/name
            folder.mkdir(parents=True)
            status['archive_sha256'] = archive_sha
            (folder/'profile-status.json').write_text(json.dumps(status))
            (folder.parent/(name+'-evidence.zip')).write_bytes(raw)
            rows.append({'group':'test','index':index,'commit':commit,
                         'receipt_sha256':operator.sha(receipt_raw),'archive_sha256':archive_sha})
        table = {'schema':'hala.h23.two_group_retention.v1','dev_experiment_sha256':operator.H22_EXPERIMENT,
                 'test_experiment_sha256':test_sha,'profiles':rows}
        def fetched(url,expected,limit):
            raw = remote[url]
            if len(raw)>limit or operator.sha(raw)!=expected: raise ValueError('CPU replay digest mismatch')
            return raw
        return table,test,test_sha,fetched

    def test_two_groups_verify_all_six_public_raws_without_old_dev_local_files(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            table,test,test_sha,fetched = self.fixtures(root)
            raw = json.dumps(table).encode()
            with patch.object(self.helper,'fetch',side_effect=fetched) as calls:
                result = operator.barrier(raw,operator.sha(raw),self.helper,test,test_sha,root)
            self.assertEqual(len(result),6)
            self.assertEqual(calls.call_count,12)
            self.assertFalse((root/'precision-attempt2').exists())
            self.assertEqual([row['local_raw_verified'] for row in result],[False]*3+[True]*3)

    def test_bad_groups_original_dev_rows_and_missing_test_local_block_barrier(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            table,test,test_sha,fetched = self.fixtures(root)
            variants = [dict(table,profiles=table['profiles'][:3]),dict(table,dev_experiment_sha256='f'*64),
                        dict(table,test_experiment_sha256=operator.H22_EXPERIMENT)]
            wrong_group = copy.deepcopy(table)
            wrong_group['profiles'][3]['group']='dev'
            variants.append(wrong_group)
            wrong_old = copy.deepcopy(table)
            wrong_old['profiles'][0]['commit']='f'*40
            variants.append(wrong_old)
            for value in variants:
                raw = json.dumps(value).encode()
                with self.subTest(value=value),patch.object(self.helper,'fetch',side_effect=fetched),self.assertRaises(ValueError):
                    operator.barrier(raw,operator.sha(raw),self.helper,test,test_sha,root)
            (root/'precision-test-restart/profile-2-nf4-test/profile-status.json').unlink()
            raw = json.dumps(table).encode()
            with patch.object(self.helper,'fetch',side_effect=fetched),self.assertRaises(ValueError):
                operator.barrier(raw,operator.sha(raw),self.helper,test,test_sha,root)

    def test_cross_group_qualification_and_forged_raw_result_claim_rejected(self):
        for options in ({'bad_test_attempt':True},{'forged_result':True}):
            with tempfile.TemporaryDirectory() as root:
                root = Path(root)
                table,test,test_sha,fetched = self.fixtures(root,**options)
                raw = json.dumps(table).encode()
                with self.subTest(options=options),patch.object(self.helper,'fetch',side_effect=fetched),self.assertRaises(ValueError):
                    operator.barrier(raw,operator.sha(raw),self.helper,test,test_sha,root)

    def test_bound_export_preserves_failure_raw_log_operator_and_incremental_result(self):
        with tempfile.TemporaryDirectory() as root:
            evidence = Path(root)/'evidence'
            evidence.mkdir()
            (evidence/'operator.py').write_bytes(b'# CPU fixture')
            (evidence/'experiment.json').write_bytes(b'{}')
            (evidence/'source-freeze.json').write_bytes(b'{}')
            (evidence/'invocation.log').write_bytes(b'raw failure\n')
            result = evidence/'actual-run/result.json'
            result.parent.mkdir()
            result.write_bytes(b'{"lifecycle":{"reload":{"state":"starting"}}}')
            report = {'status':'failed','actual_qualification':False}
            with patch('builtins.print'):
                export = operator.retain(evidence,report)
            with zipfile.ZipFile(export['archive']) as archive:
                self.assertEqual(archive.read('invocation.log'),b'raw failure\n')
                self.assertEqual(archive.read('actual-run/result.json'),result.read_bytes())
                self.assertIn('operator-status.json',archive.namelist())
            self.assertFalse(export['actual_qualification'])
            self.assertEqual(export['archive_sha256'],operator.sha(Path(export['archive']).read_bytes()))

    def test_official_timeout_owns_only_new_invocation_group_and_marks_incomplete(self):
        class Selector:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def register(self,*args): pass
            def select(self,*args): return [True]
        stdout = SimpleNamespace(fileno=lambda:123,close=lambda:None)
        process = SimpleNamespace(pid=12345,stdout=stdout,wait=lambda timeout:124)
        with tempfile.TemporaryDirectory() as root, \
                patch.object(operator.subprocess,'Popen',return_value=process) as launched, \
                patch.object(operator.selectors,'DefaultSelector',return_value=Selector()), \
                patch.object(operator.os,'read',side_effect=[b'raw timeout\n',b'']), \
                patch.object(operator.os,'killpg',side_effect=ProcessLookupError,create=True):
            report = {}
            operator.invoke(Path('/source'),Path(root),report)
            args = launched.call_args.args[0]
            self.assertEqual(args[:5],['/usr/bin/timeout','--signal=TERM','--kill-after=5s','1200s',operator.sys.executable])
            self.assertTrue(launched.call_args.kwargs['start_new_session'])
            self.assertEqual(launched.call_count,1)
            self.assertEqual(report['status'],'incomplete_timeout')
            self.assertTrue(report['all_qualification_false'])

    def test_read_selector_and_log_errors_cleanup_owned_group_before_archive(self):
        class Selector:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def register(self,*args): pass
            def select(self,*args): return [True]
        for failure in ('read','selector','log'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as root:
                stdout = unittest.mock.Mock()
                stdout.fileno.return_value=123
                process = unittest.mock.Mock(pid=12345,stdout=stdout)
                process.wait.return_value=-15
                selector = Selector()
                if failure=='selector': selector.register=unittest.mock.Mock(side_effect=OSError('selector failed'))
                signals=[]
                def killpg(pid,sig):
                    self.assertEqual(pid,12345)
                    signals.append(sig)
                    if sig==0: raise ProcessLookupError
                report={}
                opened=Path.open
                def log_open(path,*args,**kwargs):
                    if failure=='log' and path.name=='invocation.log': raise OSError('log failed')
                    return opened(path,*args,**kwargs)
                with patch.object(operator.subprocess,'Popen',return_value=process), \
                        patch.object(operator.selectors,'DefaultSelector',return_value=selector), \
                        patch.object(operator.os,'read',side_effect=OSError('read failed')), \
                        patch.object(operator.os,'killpg',side_effect=killpg,create=True), \
                        patch.object(Path,'open',log_open):
                    with self.assertRaises(OSError): operator.invoke(Path('/source'),Path(root),report)
                self.assertIn(signal.SIGTERM,signals)
                self.assertTrue(report['producer_quiescent'])
                self.assertTrue(report['cleanup']['leader_reaped'])
                process.wait.assert_called()
                stdout.close.assert_called_once()

    def test_cleanup_double_timeout_retains_error_and_blocks_normal_archive(self):
        process=unittest.mock.Mock(pid=12345)
        process.wait.side_effect=operator.subprocess.TimeoutExpired('owned-timeout',2)
        with patch.object(operator.os,'killpg',create=True) as signals, \
                patch.object(operator.signal,'SIGKILL',9,create=True), \
                patch.object(operator.time,'monotonic',side_effect=range(1000)),patch.object(operator.time,'sleep'):
            cleanup=operator.finish_owned(process,abort=True)
        self.assertFalse(cleanup['leader_reaped'])
        self.assertFalse(cleanup['confirmed'])
        self.assertEqual(signals.call_args_list[0].args,(12345,signal.SIGTERM))
        self.assertTrue(any(call.args==(12345,9) for call in signals.call_args_list))
        with tempfile.TemporaryDirectory() as root,patch('builtins.print'):
            evidence=Path(root)/'evidence'
            evidence.mkdir()
            export=operator.retain(evidence,{'status':'incomplete_cleanup','producer_quiescent':False,'cleanup':cleanup})
            self.assertIsNone(export['archive'])
            self.assertFalse(evidence.with_suffix('.zip').exists())
            self.assertEqual(export['retention_blocked'],'owned producer not confirmed quiescent')


if __name__=='__main__': unittest.main()
