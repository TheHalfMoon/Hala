import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from development.data import build_dataset, validate_dataset
from development.protocol import parse_response
from development.simulator import Simulator, oracle, score_trace
from development.verifier import verify_final
from development.media import render_frame
from development.freeze import freeze_bundle, verify_bundle
from development.runner import run_episode, run_mind_cases


class DevelopmentTests(unittest.TestCase):
    def test_disjoint_bilingual_data(self):
        data = build_dataset()
        validate_dataset(data)
        self.assertEqual(len(data['episodes']), 32)
        self.assertEqual(len(data['rows']), 136)
        self.assertEqual({r['language'] for r in data['rows']}, {'en', 'ar'})
        families = [{e['family'] for e in data['episodes'] if e['split'] == split}
                    for split in ('train', 'dev', 'test')]
        self.assertFalse(families[0] & families[1] | families[0] & families[2] | families[1] & families[2])

    def test_all_oracles_really_finish_and_corruption_fails(self):
        for episode in build_dataset()['episodes']:
            sim = Simulator(episode)
            self.assertFalse(verify_final(episode, sim.snapshot()))
            while not verify_final(episode, sim.snapshot()):
                sim.apply(oracle(sim))
            self.assertTrue(verify_final(episode, sim.snapshot()))
            broken = sim.snapshot()
            broken['saved_value'] = 'wrong'
            self.assertFalse(verify_final(episode, broken))
            self.assertTrue(score_trace(episode, sim.trace, sim.snapshot())['success'])

    def test_parser_rejects_ambiguous_and_unsafe(self):
        bad = ['{"action":"click","point":[true,2]}',
               '{"action":"click","point":[0,1001]}',
               '{"action":"click","point":[1,2],"authority":true}',
               '{"action":"click","action":"click","point":[1,2]}',
               '{"action":"shell","text":"echo bad"}',
               '{"action":"type","point":[1,2],"text":"x","extra":0}',
               '{"action":"click","point":[NaN,2]}', '```json\n{}\n```']
        for value in bad:
            with self.assertRaises(ValueError, msg=value):
                parse_response('type', value)

    def test_wrong_order_proposal_and_false_receipt_do_not_finish(self):
        episode = build_dataset()['episodes'][0]
        sim = Simulator(episode)
        sim.apply({'action': 'click', 'point': [800, 850]})
        sim.apply({'action': 'shell', 'text': 'fake receipt'})
        metrics = score_trace(episode, sim.trace, sim.snapshot())
        self.assertFalse(metrics['success'])
        self.assertEqual(metrics['unsafe_proposals'], 1)
        self.assertGreater(metrics['action_errors'], 0)

    def test_independent_verifier_does_not_accept_claim(self):
        episode = build_dataset()['episodes'][0]
        self.assertFalse(verify_final(episode, {'success': True}))
        with self.assertRaises(ValueError):
            parse_response('mind', '{"kind":"plan","steps":["execute shell"]}')

    def test_dataset_corruption(self):
        for field, value in [('completion', '{"route":"abstain"}'), ('split', 'test')]:
            data = copy.deepcopy(build_dataset())
            data['rows'][0][field] = value
            with self.assertRaises(ValueError):
                validate_dataset(data)

    def test_prompt_no_private_labels(self):
        for row in build_dataset()['rows']:
            prompt = row['prompt']
            for forbidden in ('expected_state', 'target_box', 'episode_id', 'fixture_id'):
                self.assertNotIn(forbidden, prompt)

    def test_standalone_mind_all_categories_both_languages(self):
        cases = build_dataset()['mind_cases']
        self.assertEqual(len(cases), 10)
        self.assertEqual({c['category'] for c in cases}, {'plan', 'visual', 'observe', 'clarify', 'abstain'})
        expected = iter(cases)
        results = run_mind_cases(cases, lambda *args: json.dumps(next(expected)['expected']))
        self.assertTrue(all(r['correct'] for r in results))

    def test_arabic_media_fail_closed_on_unqualified_host(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(ValueError):
                render_frame(build_dataset()['episodes'][0], Simulator(build_dataset()['episodes'][0]).snapshot(),
                             Path(path) / 'image.png', Path(path) / 'missing.ttf')

    def test_freeze_no_overwrite_or_false_media_claim(self):
        with tempfile.TemporaryDirectory() as path:
            output = Path(path) / 'freeze'
            result = freeze_bundle(output, {'seed': 42})
            self.assertFalse(result['media_qualified'])
            verify_bundle(output, require_media=False)
            with self.assertRaises(ValueError):
                verify_bundle(output)
            with self.assertRaises(ValueError):
                freeze_bundle(output, {'seed': 42})
            (output / 'dataset.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                verify_bundle(output, require_media=False)

    def test_closed_loop_runner_uses_new_observation_and_separate_mind(self):
        episode = next(e for e in build_dataset()['episodes'] if e['split'] == 'test' and e['transient_failure'])
        def infer(component, text, image):
            content = json.loads(text[text.index('{"goal"'):])
            observation = content['observation']
            if component == 'mind':
                self.assertIsNone(image)
                if content['request'] == 'plan':
                    from development.data import plan_label
                    return json.dumps(plan_label())
                tool = 'observe' if 'temporarily failed' in observation['status'] else 'visual'
                return json.dumps({'kind': 'next', 'tool': tool, 'subgoal': 'inspect_failure' if tool == 'observe' else 'replace_field'})
            self.assertIsNotNone(image)
            from development.simulator import center
            if observation['field_value'] != episode['requested_value']:
                return json.dumps({'action': 'type', 'point': center('field', episode), 'text': episode['requested_value']})
            if observation['item_location'] != episode['destination']:
                return json.dumps({'action': 'drag', 'start': center('item', episode), 'end': center('destination', episode)})
            return json.dumps({'action': 'click', 'point': center('save', episode)})
        with tempfile.TemporaryDirectory() as path, patch('development.runner.render_frame', return_value={'mock': True}):
            result = run_episode(episode, infer, path, 'unused')
        self.assertTrue(result['metrics']['success'])
        self.assertEqual(result['metrics']['recoveries'], 1)
        self.assertEqual(result['metrics']['action_errors'], 0)
        self.assertEqual(result['metrics']['environment_errors'], 1)
        self.assertEqual(result['metrics']['model_calls'], 9)
        self.assertFalse(result['native_qualified'])

    def test_layouts_are_distinct_and_rtl_mirrors(self):
        from development.simulator import layout, WIDTH
        episodes = build_dataset()['episodes']
        english = [e for e in episodes if e['language'] == 'en']
        self.assertEqual(len({json.dumps(layout(e), sort_keys=True) for e in english}), 16)
        for en in english:
            ar = next(e for e in episodes if e['family'] == en['family'] and e['variant'] == en['variant'] and e['language'] == 'ar')
            for name, (x, y, w, h) in layout(en).items():
                self.assertEqual(layout(ar)[name], (WIDTH-x-w, y, w, h))

    def test_redundant_observations_consume_budget(self):
        episode = build_dataset()['episodes'][0]
        result = run_episode(episode, lambda *args: '{"kind":"next","tool":"observe","subgoal":"verify_save"}', '.', 'unused')
        self.assertFalse(result['metrics']['success'])
        self.assertEqual(result['metrics']['unnecessary_calls'], 7)
        self.assertFalse(result['calls'][0]['requested_kind_correct'])

    def test_old_observation_cannot_satisfy_new_failure(self):
        from development.simulator import center
        episode = next(e for e in build_dataset()['episodes'] if e['transient_failure'])
        sim = Simulator(episode)
        sim.observe()
        sim.apply(oracle(sim))
        sim.apply(oracle(sim))
        save = {'action': 'click', 'point': center('save', episode)}
        self.assertTrue(sim.apply(save)['environment_error'])
        self.assertFalse(sim.snapshot()['observed'])
        self.assertTrue(sim.apply(save)['action_error'])
        self.assertFalse(verify_final(episode, sim.snapshot()))
        sim.observe()
        self.assertTrue(sim.apply(save)['recovered'])
        self.assertTrue(verify_final(episode, sim.snapshot()))

    def test_wrong_mind_decomposition_cannot_score_as_correct(self):
        from development.simulator import center
        episode = build_dataset()['episodes'][0]
        def infer(component, text, image):
            observation = json.loads(text[text.index('{"goal"'):])['observation']
            if component == 'mind':
                return '{"kind":"next","tool":"visual","subgoal":"save_changes"}'
            if observation['field_value'] != episode['requested_value']:
                return json.dumps({'action': 'type', 'point': center('field', episode), 'text': episode['requested_value']})
            if observation['item_location'] != episode['destination']:
                return json.dumps({'action': 'drag', 'start': center('item', episode), 'end': center('destination', episode)})
            return json.dumps({'action': 'click', 'point': center('save', episode)})
        with patch('development.runner.render_frame', return_value={'mock': True}):
            result = run_episode(episode, infer, '.', 'unused')
        self.assertTrue(result['metrics']['success'])
        self.assertGreaterEqual(result['metrics']['mind_errors'], 3)
        self.assertEqual(result['metrics']['plan_accuracy'], 0)
        self.assertFalse(result['calls'][0]['requested_kind_correct'])

    def test_independent_mind_semantics_tracks_bilingual_state(self):
        from development.data import plan_label, next_label
        from development.verifier import verify_mind_semantics
        for episode in build_dataset()['episodes']:
            sim = Simulator(episode)
            self.assertTrue(verify_mind_semantics(episode['goal'], sim.observation(), plan_label(), 'plan'))
            while not verify_final(episode, sim.snapshot()):
                label = next_label(sim)
                self.assertTrue(verify_mind_semantics(episode['goal'], sim.observation(), label, 'next'))
                wrong = dict(label, subgoal='save_changes' if label['subgoal'] != 'save_changes' else 'replace_field')
                self.assertFalse(verify_mind_semantics(episode['goal'], sim.observation(), wrong, 'next'))
                sim.apply(oracle(sim))


if __name__ == '__main__':
    unittest.main()
