"""Adversarial small-adapter archival tests, without model dependencies."""
import hashlib
import ast
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from training.adapter_archive import create_bundle, verify_bundle, EXPECTED_SHAPES, TARGET_PATTERN


BASE = 'Qwen/Qwen3-VL-2B-Instruct'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
IDENTITY = {'base_model': BASE, 'base_revision': REVISION,
            'source_commit': 'a' * 40, 'data_sha256': 'b' * 64,
            'evaluation_sha256': 'c' * 64}


def tensor_bytes():
    header = {}
    offset = 0
    for layer in range(28):
        for module, output_size in [('q_proj', 2048), ('v_proj', 1024)]:
            for suffix, shape in [('A', [4, 2048]), ('B', [output_size, 4])]:
                size = shape[0] * shape[1] * 4
                header[f'base_model.model.model.language_model.layers.{layer}.self_attn.{module}.lora_{suffix}.weight'] = {
                    'dtype': 'F32', 'shape': shape, 'data_offsets': [offset, offset + size]}
                offset += size
    encoded = json.dumps(header).encode()
    return struct.pack('<Q', len(encoded)) + encoded + bytes(offset)


def producer_config():
    """PEFT0.20 dataclass defaults plus the actual qualified producer overrides.

    Official wheel SHA256: 0fbba16ffebfad3de96e06f2da6860fd860292324b85b6141909fa1e26ea9233.
    PEFT save removes runtime_config; version is autofilled and inference mode set.
    """
    value = dict.fromkeys(['auto_mapping', 'exclude_modules', 'modules_to_save',
        'layers_to_transform', 'layers_pattern', 'megatron_config', 'trainable_token_indices',
        'eva_config', 'corda_config', 'lora_ga_config', 'velora_config', 'alora_invocation_tokens',
        'monteclora_config', 'layer_replication', 'target_parameters', 'use_bdlora', 'arrow_config'])
    value.update(base_model_name_or_path=BASE, revision=None, peft_type='LORA', task_type='CAUSAL_LM',
        peft_version='0.20.0', inference_mode=True, r=4,
        target_modules=r'.*language_model.*\.(q_proj|v_proj)$', lora_alpha=8, lora_dropout=0,
        fan_in_fan_out=False, bias='none', use_rslora=False, init_lora_weights=True,
        rank_pattern={}, alpha_pattern={}, megatron_core='megatron.core', loftq_config={},
        use_dora=False, use_qalora=False, qalora_group_size=16, lora_bias=False,
        ensure_weight_tying=False)
    return value


class AdapterArchiveTests(unittest.TestCase):
    def test_projection_shapes_and_producer_regex_match_pinned_source(self):
        repo = Path(__file__).resolve().parents[1]
        raw = (repo / 'research/evidence/Qwen--Qwen3-VL-2B-Instruct/config.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         'bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b')
        config = json.loads(raw)['text_config']
        calculated = {}
        for layer in range(config['num_hidden_layers']):
            for module, heads in [('q_proj', config['num_attention_heads']),
                                  ('v_proj', config['num_key_value_heads'])]:
                for suffix, shape in [('A', [4, config['hidden_size']]),
                                      ('B', [heads * config['head_dim'], 4])]:
                    calculated[f'base_model.model.model.language_model.layers.{layer}.self_attn.{module}.lora_{suffix}.weight'] = shape
        self.assertEqual(EXPECTED_SHAPES, calculated)
        tree = ast.parse((repo / 'training/scripts/train_adapter.py').read_text())
        producer = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == 'LORA_TARGET_PATTERN'
                            for target in node.targets))
        self.assertEqual(TARGET_PATTERN, producer)

    def fixture(self, root):
        source = root / 'source'
        source.mkdir()
        (source / 'adapter_model.safetensors').write_bytes(tensor_bytes())
        (source / 'adapter_config.json').write_text(json.dumps({
            'base_model_name_or_path': BASE, 'revision': REVISION,
            'peft_type': 'LORA', 'peft_version': '0.20.0', 'task_type': 'CAUSAL_LM', 'r': 4,
            'target_modules': ['q_proj', 'v_proj'], 'bias': 'none',
            'lora_alpha': 8, 'lora_dropout': 0.0}), encoding='utf-8')
        return source

    def test_exact_official_peft_producer_config_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            config = source / 'adapter_config.json'
            raw = json.dumps(producer_config(), indent=2).encode()
            config.write_bytes(raw)
            digest = create_bundle(source, root / 'bundle', IDENTITY)
            verify_bundle(root / 'bundle', IDENTITY, digest)
            self.assertEqual((root / 'bundle' / config.name).read_bytes(), raw)
            for changes in [{'target_modules': '.*(q_proj|v_proj)$'}, {'peft_version': '0.21.0'},
                            {'use_bdlora': {}}, {'lora_ga_config': {}}, {'velora_config': {}},
                            {'monteclora_config': {}}, {'init_lora_weights': 'pissa'}]:
                config.write_text(json.dumps(dict(producer_config(), **changes)))
                with self.assertRaises(ValueError):
                    create_bundle(source, root / 'invalid', IDENTITY)

    def test_complete_exact_pinned_tensor_inventory_required(self):
        good = tensor_bytes()
        size = struct.unpack('<Q', good[:8])[0]
        header = json.loads(good[8:size + 8])
        self.assertEqual(len(header), 112)
        self.assertEqual(sum(v['shape'][0] * v['shape'][1] for v in header.values()), 802816)
        for transform in ['partial', 'layer99', 'wrong_shape', 'alternate_prefix']:
            changed = json.loads(json.dumps(header))
            if transform == 'partial':
                # Preserve exact contiguous bytes for one valid q A/B pair.
                changed = dict(list(changed.items())[:2])
                payload = good[size + 8:size + 8 + list(changed.values())[-1]['data_offsets'][1]]
            else:
                payload = good[size + 8:]
                key = next(iter(changed))
                if transform == 'wrong_shape':
                    changed[key]['shape'] = [4, 1024]
                else:
                    new = key.replace('layers.0.', 'layers.99.') if transform == 'layer99' else key.replace('model.model.model.', 'model.model.')
                    changed[new] = changed.pop(key)
            raw_header = json.dumps(changed).encode()
            with self.subTest(transform=transform), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = self.fixture(root)
                (source / 'adapter_model.safetensors').write_bytes(struct.pack('<Q', len(raw_header)) + raw_header + payload)
                with self.assertRaises(ValueError):
                    create_bundle(source, root / 'bundle', IDENTITY)

    @unittest.skipUnless(os.name == 'nt', 'Windows junction regression')
    def test_real_junction_rejected_when_pathlib_api_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            link = root / 'redirect'
            script = "New-Item -ItemType Junction -Path '" + str(link).replace("'", "''") + "' -Target '" + str(source).replace("'", "''") + "' | Out-Null"
            subprocess.run(['rtk', 'proxy', 'powershell', '-NoProfile', '-Command', script], check=True, capture_output=True)
            saved = getattr(Path, 'is_junction', None)
            try:
                if saved is not None:
                    delattr(Path, 'is_junction')
                with self.assertRaises(ValueError):
                    create_bundle(link, root / 'bundle', IDENTITY)
            finally:
                if saved is not None:
                    Path.is_junction = saved
                os.rmdir(link)

    def test_deterministic_roundtrip_and_external_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            first = create_bundle(source, root / 'one', IDENTITY)
            second = create_bundle(source, root / 'two', IDENTITY)
            self.assertEqual(first, second)
            result = verify_bundle(root / 'one', IDENTITY, first)
            self.assertTrue(result['contains_adapter'])
            self.assertFalse(result['quality_qualified'])
            with self.assertRaises(ValueError):
                verify_bundle(root / 'one', dict(IDENTITY, source_commit='d' * 40), first)

    def test_pretraining_roundtrip_cannot_be_mistaken_for_asset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'empty'
            source.mkdir()
            digest = create_bundle(source, root / 'bundle', IDENTITY, kind='pretraining_roundtrip')
            result = verify_bundle(root / 'bundle', IDENTITY, digest)
            self.assertFalse(result['contains_adapter'])
            self.assertEqual(result['files'], {})

    def test_extra_pickle_and_nested_files_rejected(self):
        for name in ['optimizer.pt', 'README.md', 'nested']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = self.fixture(root)
                (source / name).write_bytes(b'unsafe')
                with self.assertRaises(ValueError):
                    create_bundle(source, root / 'bundle', IDENTITY)
    def test_real_symlink_files_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            original = source / 'adapter_model.safetensors'
            target = root / 'external'
            original.replace(target)
            try:
                original.symlink_to(target)
            except OSError:
                self.skipTest('Symlinks unavailable to this Windows account')
            with self.assertRaises(ValueError):
                create_bundle(source, root / 'bundle', IDENTITY)

    def test_symlink_component_rejection_without_windows_privileges(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            with patch.object(Path, 'is_symlink', return_value=True):
                with self.assertRaises(ValueError):
                    create_bundle(source, root / 'bundle', IDENTITY)

    def test_untrusted_metadata_and_manifest_types_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            metadata = source / 'training_metadata.json'
            metadata.write_text('{"credential":"secret"}')
            with self.assertRaises(ValueError):
                create_bundle(source, root / 'bundle', IDENTITY)
            metadata.unlink()
            digest = create_bundle(source, root / 'bundle', IDENTITY)
            bundle = root / 'bundle'
            original = json.loads((bundle / 'manifest.json').read_text())
            for mutation in [{'kind': []}, {'contains_adapter': 1}, {'quality_qualified': True},
                             {'identity': dict(IDENTITY, data_sha256='d' * 64)},
                             {'files': {'../adapter_model.safetensors': {}}}]:
                value = dict(original, **mutation)
                raw = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
                (bundle / 'manifest.json').write_bytes(raw)
                with self.assertRaises(ValueError):
                    verify_bundle(bundle, IDENTITY, hashlib.sha256(raw).hexdigest())

    def test_malformed_tensor_headers_ranges_and_base_weights(self):
        good = tensor_bytes()
        length = struct.unpack('<Q', good[:8])[0]
        header = json.loads(good[8:8 + length])
        mutations = []
        for field, value in [('dtype', 'I64'), ('shape', [True, 2]),
                             ('data_offsets', [0, 99999])]:
            mutated = json.loads(json.dumps(header))
            mutated[next(iter(mutated))][field] = value
            mutations.append(mutated)
        mutations.append({'base_model.weight': next(iter(header.values()))})
        mutated = json.loads(json.dumps(header))
        mutated[list(mutated)[1]]['data_offsets'] = [0, 32]
        mutations.append(mutated)
        for bad in [b'', struct.pack('<Q', 2 ** 63), *[
                struct.pack('<Q', len(encoded)) + encoded + bytes(64)
                for encoded in [json.dumps(item).encode() for item in mutations]]]:
            with self.subTest(length=len(bad)), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = self.fixture(root)
                (source / 'adapter_model.safetensors').write_bytes(bad)
                with self.assertRaises(ValueError):
                    create_bundle(source, root / 'bundle', IDENTITY)

    def test_hash_tamper_manifest_duplicate_and_extra_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            bundle = root / 'bundle'
            digest = create_bundle(source, bundle, IDENTITY)
            tensor = bundle / 'adapter_model.safetensors'
            tensor.write_bytes(tensor.read_bytes()[:-1] + b'X')
            with self.assertRaises(ValueError):
                verify_bundle(bundle, IDENTITY, digest)
            tensor.write_bytes((source / tensor.name).read_bytes())
            (bundle / 'unexpected').write_bytes(b'x')
            with self.assertRaises(ValueError):
                verify_bundle(bundle, IDENTITY, digest)
            (bundle / 'unexpected').unlink()
            path = bundle / 'manifest.json'
            path.write_bytes(b'{"schema":1,"schema":1}')
            with self.assertRaises(ValueError):
                verify_bundle(bundle, IDENTITY, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_wrong_config_and_oversize_or_nonimmutable_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.fixture(root)
            config = source / 'adapter_config.json'
            original = json.loads(config.read_text())
            for updates in [{'base_model_name_or_path': 'other'}, {'auto_mapping': {'path': 'evil'}},
                            {'target_modules': ['visual']}, {'modules_to_save': ['lm_head']}]:
                config.write_text(json.dumps(dict(original, **updates)))
                with self.assertRaises(ValueError):
                    create_bundle(source, root / 'bundle', IDENTITY)
            config.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                create_bundle(source, root / 'bundle', dict(IDENTITY, source_commit='main'))
            with open(source / 'adapter_model.safetensors', 'wb') as handle:
                handle.truncate(8 * 1024 * 1024 + 1)
            with self.assertRaises(ValueError):
                create_bundle(source, root / 'bundle', IDENTITY)


if __name__ == '__main__':
    unittest.main()
