"""TEST-only restart operator: retain all original DEV, then run one remaining TEST profile."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path('/kaggle/working/hala-h21')
COMMIT = '07591f109b6683c9302fcf890259ddf0bde91494'
FREEZE = '98ca7bba6fc16dd6cd737eaad6de1e425e30aef67116c379db36fbbb85b2af4b'
ORDER = [['fp16', 'test'], ['int8', 'test'], ['nf4', 'test']]
DEV_ORDER = [['fp16', 'dev'], ['int8', 'dev'], ['nf4', 'dev']]
DEV_SHA = '60a0986cc196041c2474314624e0f88c9c2986d75bc58cf0ddb44623ed294e02'
DEV_RETENTION = [
    {'index': 0, 'commit': 'e252b7805055fa6a89e1d92b6bba7faa5dbf1efc',
     'receipt_sha256': '0664e9c17358a3ab38cc6ba3df04852ca7e6b59a027117725502c3fe70f6531a',
     'archive_sha256': '6d5cfe7c3746a76eaceb8ebc37605d0d98cc69b69ca94765421f081b64d64580'},
    {'index': 1, 'commit': 'e3d400424766aad3c5856ba29266e11033392ef5',
     'receipt_sha256': 'd693bdf0cb5fede0931cb423296efbf0a7a470b4a91039c1b23ff90089936af7',
     'archive_sha256': 'e0dbe2d692c522943b3e0c49c198e540416cbabfa2b19499c9da4a518b8339e5'},
    {'index': 2, 'commit': 'd08f522de433d9da866ec3907161e4a9b326efba',
     'receipt_sha256': '641c760bd286991c5c5d565cd8a1b3034a747e37cbcb109af5d0da8a4495a850',
     'archive_sha256': '84dac2fd6ab44d751cefbe9ce61a958ef44a39aad3b1f09e880c5350bbd87458'}]
PREFIX = 'https://raw.githubusercontent.com/TheHalfMoon/Hala/'
ARTIFACT = 'h22-test-restart/'
ORIGINAL_SHA = '3742fc46ec9f4e5151e24f20a78450ab315c25aaae70b002408a64af44a87f39'
ORIGINAL_URL = PREFIX + '253dccf22f54ca0aead4afc3137da1a2276276a0/h22-prospective-20261016/experiment.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def hex_digest(value, length=64):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is None:
        raise ValueError('Immutable lowercase digest required')


def artifact_url(commit, name, *, group='test'):
    hex_digest(commit, 40)
    if group not in ('test', 'dev'):
        raise ValueError('Fixed DEV or TEST retention group required')
    return PREFIX + commit + '/' + (ARTIFACT if group == 'test' else 'h22-attempt2/') + name


def fetch(url, expected, limit):
    hex_digest(expected)
    # Redirects must not turn this public immutable evidence read into another host.
    with urllib.request.urlopen(url, timeout=45) as response:
        if response.geturl() != url:
            raise ValueError('Evidence URL redirected')
        raw = response.read(limit + 1)
    if len(raw) > limit or sha(raw) != expected:
        raise ValueError('Bounded public artifact digest mismatch')
    return raw


def clean_source(root=ROOT):
    source = root / 'source'
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True, timeout=30).strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=source,
                           capture_output=True, text=True, check=True, timeout=30).stdout
    if actual != COMMIT or dirty.strip():
        raise ValueError('Exact complete clean published source required')


def profile_name(index):
    if type(index) is not int or not 0 <= index < 3:
        raise ValueError('Exactly one fixed TEST-only index0..2 required')
    return 'profile-' + str(index) + '-' + '-'.join(ORDER[index])


def validate_manifest(manifest, original):
    # Only attempt hardware/cache/retention metadata may differ from the original.
    fixed = ('source_commit', 'source_freeze_sha256', 'h19_manifest_sha256', 'adapter_manifest_sha256',
             'adapter_identity', 'packages', 'budget', 'cache', 'model_choice', 'precision_scope')
    for key in fixed:
        if json.dumps(manifest[key], sort_keys=True) != json.dumps(original[key], sort_keys=True):
            raise ValueError('Original study setting changed: ' + key)
    if (manifest['source_commit'] != COMMIT or manifest['source_freeze_sha256'] != FREEZE or
            manifest['mode_order'] != ORDER or manifest.get('attempt') != 3 or
            manifest.get('dev_experiment_sha256') != DEV_SHA or manifest.get('prior_dev_results_retained') is not True or
            manifest.get('dev_retention') != DEV_RETENTION or
            manifest.get('adoption_qualified') is not False or manifest.get('quality_qualified') is not False or
            re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', manifest['gpu_uuid']) is None):
        raise ValueError('TEST-only prospective amendment identity required')


def prior_retention(index, experiment_sha, artifact_commit, receipt_sha, archive_sha, *, group='test'):
    if group not in ('test', 'dev') or type(index) is not int or not 1 <= index <= 3:
        raise ValueError('Fixed retained group and index required')
    if group == 'dev' and experiment_sha != DEV_SHA:
        raise ValueError('Original retained DEV experiment required')
    order = ORDER if group == 'test' else DEV_ORDER
    prefix = ARTIFACT if group == 'test' else 'h22-attempt2/'
    name = 'profile-' + str(index - 1) + '-' + '-'.join(order[index - 1])
    receipt = json.loads(fetch(artifact_url(artifact_commit, name + '-receipt.json', group=group), receipt_sha, 65536))
    expected = {'schema': 'hala.h22.public_retention_receipt.v1', 'index': index - 1,
                'experiment_sha256': experiment_sha, 'archive_sha256': archive_sha,
                'archive_path': prefix + name + '-evidence.zip'}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Prior receipt index/experiment/archive binding mismatch')
    raw = fetch(artifact_url(artifact_commit, name + '-evidence.zip', group=group), archive_sha, 16 * 1024**2)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or len(infos) > 1000 or sum(i.file_size for i in infos) > 64 * 1024**2:
            raise ValueError('Bounded unique prior ZIP inventory required')
        for item in infos:
            if item.file_size > 8 * 1024**2 or item.filename.startswith('/') or '..' in Path(item.filename).parts:
                raise ValueError('Invalid prior ZIP member')
        required = {'experiment.json', 'profile-status.json', 'profile.log', 'qualification.json', 'operator-launcher.py'}
        if not required.issubset(names) or not archive.read('operator-launcher.py'):
            raise ValueError('Prior raw log/qualification/launcher evidence missing')
        if sha(archive.read('experiment.json')) != experiment_sha:
            raise ValueError('Prior raw ZIP experiment mismatch')
        experiment = json.loads(archive.read('experiment.json'))
        qualification = json.loads(archive.read('qualification.json'))
        if any(qualification.get(key) != value for key, value in {'attempt': 3 if group == 'test' else 2, 'source_commit': COMMIT,
                'source_freeze_sha256': FREEZE, 'gpu_uuid': experiment['gpu_uuid']}.items()):
            raise ValueError('Prior raw qualification identity mismatch')
        archive.read('profile.log')  # Read/CRC-check the retained raw log, including genuine empty logs.
        previous = json.loads(archive.read('profile-status.json'))
        if (type(previous.get('index')) is not int or previous['index'] != index - 1 or
                previous.get('experiment_sha256') != experiment_sha or previous.get('name') != name or
                previous.get('source_commit') != COMMIT or previous.get('schema') != 'hala.h22.single_profile.v1'):
            raise ValueError('Prior raw ZIP profile mismatch')
        argv = previous.get('argv', [])
        for option, expected_value in (('--mode', order[index - 1][0]), ('--split', order[index - 1][1])):
            if (not isinstance(argv, list) or argv.count(option) != 1 or
                    argv.index(option) + 1 >= len(argv) or argv[argv.index(option) + 1] != expected_value):
                raise ValueError('Prior raw profile mode/allocation mismatch')
        if previous.get('status') not in ('finished', 'failed', 'launcher_timeout', 'launcher_error', 'physical_device_identity_mismatch'):
            raise ValueError('Prior profile must have a terminal retained outcome')
        claimed = previous.get('result_sha256')
        if claimed is not None or 'result_status' in previous or 'run/result.json' in names or previous['status'] == 'finished':
            if claimed is None or 'run/result.json' not in names:
                raise ValueError('Claimed/successful result must retain actual raw result JSON')
            hex_digest(claimed)
            result_raw = archive.read('run/result.json')
            if sha(result_raw) != claimed:
                raise ValueError('Prior raw result SHA256 mismatch')
            result = json.loads(result_raw)
            if (result.get('status') != previous.get('result_status') or result.get('source_commit') != COMMIT or
                    result.get('split') != order[index - 1][1] or
                    result.get('precision_profile', {}).get('mode') != order[index - 1][0] or
                    result.get('precision_profile', {}).get('source_freeze_sha256') != FREEZE):
                raise ValueError('Prior raw result status/profile identity mismatch')
            if previous['status'] == 'finished' and (previous.get('returncode') != 0 or result['status'] != 'measured' or
                    result.get('gpu_uuid') != experiment['gpu_uuid']):
                raise ValueError('Successful measured status requires matching raw measured result')
    return receipt


def dev_prerequisites():
    """The first remaining TEST profile requires all three original public raw DEV ZIPs."""
    return [prior_retention(row['index'] + 1, DEV_SHA, row['commit'], row['receipt_sha256'],
                            row['archive_sha256'], group='dev') for row in DEV_RETENTION]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run(index, manifest_commit, manifest_sha, *, previous_commit=None, receipt_sha=None, archive_sha=None, root=ROOT):
    name = profile_name(index)
    clean_source(root)
    raw = fetch(artifact_url(manifest_commit, 'experiment.json'), manifest_sha, 65536)
    manifest = json.loads(raw)
    original = json.loads(fetch(ORIGINAL_URL, ORIGINAL_SHA, 65536))
    validate_manifest(manifest, original)
    qualified = json.loads((root / 'h22-test-restart-qualification.json').read_text())
    if any(qualified.get(key) != value for key, value in {'source_commit': COMMIT,
            'source_freeze_sha256': FREEZE, 'gpu_uuid': manifest['gpu_uuid'], 'attempt': 3}.items()):
        raise ValueError('Actual TEST-only qualification identity mismatch')
    if index > 0:
        if any(value is None for value in (previous_commit, receipt_sha, archive_sha)):
            raise ValueError('Public prior raw ZIP receipt and archive hashes required before next profile')
        receipt = prior_retention(index, manifest_sha, previous_commit, receipt_sha, archive_sha)
    else:
        if any(value is not None for value in (previous_commit, receipt_sha, archive_sha)):
            raise ValueError('First profile has no previous retention receipt')
        receipt = None
    dev_receipts = dev_prerequisites() if index == 0 else None
    clean_source(root)
    results = root / 'precision-test-restart'
    results.mkdir(exist_ok=True)
    evidence = results / name
    evidence.mkdir(exist_ok=False)  # Never rerun or replace a profile, including failed attempts.
    (evidence / 'experiment.json').write_bytes(raw)
    if receipt is not None:
        atomic_json(evidence / 'previous-retention-receipt.json', receipt)
    if dev_receipts is not None:
        atomic_json(evidence / 'dev-retention-receipts.json', dev_receipts)
    mode, split = ORDER[index]
    output = evidence / 'run'
    args = [sys.executable, '-u', '-m', 'evaluation.development_precision', '--mode', mode,
        '--source-freeze-sha256', FREEZE, '--bundle', str(root / 'frozen'),
        '--manifest-sha256', manifest['h19_manifest_sha256'], '--output', str(output),
        '--font-path', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '--source-commit', COMMIT,
        '--provider', 'kaggle', '--split', split, '--adapter', str(root / 'retained-adapter'),
        '--adapter-manifest-sha256', manifest['adapter_manifest_sha256'],
        '--adapter-identity', json.dumps(manifest['adapter_identity'], sort_keys=True)]
    env = dict(os.environ, HALA_REMOTE_DEVELOPMENT_ALLOWED='1', PYTHONIOENCODING='utf-8',
        HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key in ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE', 'XDG_CACHE_HOME', 'TORCH_HOME'):
        env[key] = '/tmp/hala-h22-cache'
    report = {'schema': 'hala.h22.single_profile.v1', 'index': index, 'name': name,
        'experiment_sha256': manifest_sha, 'source_commit': COMMIT, 'status': 'running',
        'started_unix': time.time(), 'argv': args, 'training_runs': 0, 'adoption_qualified': False}
    status = evidence / 'profile-status.json'
    atomic_json(status, report)
    try:
        with (evidence / 'profile.log').open('wb') as stream:
            try:
                process = subprocess.run(args, cwd=root / 'source', env=env, stdout=stream,
                    stderr=subprocess.STDOUT, timeout=3700)
                report.update(returncode=process.returncode, status='finished' if process.returncode == 0 else 'failed')
            except subprocess.TimeoutExpired:
                report.update(returncode=None, status='launcher_timeout')
            except Exception as error:
                report.update(returncode=None, status='launcher_error', error_type=type(error).__name__)
                stream.write((type(error).__name__ + '\n').encode())
        result_path = output / 'result.json'
        if result_path.is_file():
            result_raw = result_path.read_bytes()
            result = json.loads(result_raw)
            report.update(result_sha256=sha(result_raw), result_status=result['status'], summary=result.get('summary'),
                          gpu_uuid=result.get('gpu_uuid'))
            if result.get('gpu_uuid') != manifest['gpu_uuid']:
                report['status'] = 'physical_device_identity_mismatch'
    except Exception as error:
        report.update(status='launcher_error', evidence_error=type(error).__name__)
    finally:
        report['finished_unix'] = time.time()
        atomic_json(status, report)
        archive_path = results / (name + '-evidence.zip')
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(evidence.rglob('*')):
                if path.is_file():
                    if path.is_symlink() or path.stat().st_size > 8 * 1024**2:
                        raise ValueError('Evidence file bound exceeded; raw evidence retained on disk')
                    archive.write(path, str(path.relative_to(evidence)))
            archive.write(root / 'h22-test-restart-qualification.json', 'qualification.json')
            prewarm = root / 'h22-test-restart-prewarm.json'
            if prewarm.exists():
                if prewarm.is_symlink() or prewarm.stat().st_size > 65536:
                    raise ValueError('Bounded separate prewarm record required')
                archive.write(prewarm, 'prewarm.json')
            archive.write(Path(__file__), 'operator-launcher.py')
        if archive_path.stat().st_size > 16 * 1024**2:
            raise ValueError('Evidence ZIP bound exceeded; raw evidence retained on disk')
        report.update(archive_sha256=sha(archive_path.read_bytes()), archive_bytes=archive_path.stat().st_size,
                      archive_path=ARTIFACT + archive_path.name)
        atomic_json(status, report)
    print('HALA_H22_ONE_PROFILE_COMPLETE:' + json.dumps(report, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, required=True, choices=range(3))
    parser.add_argument('--manifest-commit', required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--previous-commit')
    parser.add_argument('--receipt-sha')
    parser.add_argument('--archive-sha')
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
