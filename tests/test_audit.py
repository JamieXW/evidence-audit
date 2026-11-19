import copy
import re
import unittest

from audit import answer, build, evaluate, oracle, valid_world, verify, worlds


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public, cls.keys = build()

    def missing(self):
        return next(k for k in self.keys if k['variant'] == 'missing')

    def test_counts_and_reproducibility(self):
        self.assertEqual((self.public, self.keys), build())
        self.assertEqual(len(self.keys), 80)
        self.assertEqual(len({k['scenario'] for k in self.keys}), 20)
        self.assertEqual(len({k['id'] for k in self.keys}), 80)

    def test_family_and_scenario_split(self):
        for field in ('scenario', 'family'):
            dev = {k[field] for k in self.keys if k['split'] == 'dev'}
            test = {k[field] for k in self.keys if k['split'] == 'test'}
            self.assertFalse(dev & test)

    def test_public_fields(self):
        for task in self.public:
            self.assertEqual(set(task), {'id', 'question', 'documents', 'world_schema'})

    def test_all_labels_and_truth_consistent(self):
        for row in self.keys:
            spec = row['spec']
            self.assertTrue(valid_world(spec, row['hidden_world']))
            self.assertEqual(answer(spec, row['hidden_world']), row['expected_answer'])
            possible = {answer(spec, w) for w in worlds(spec)}
            self.assertEqual(len(possible), 2 if row['variant'] in ('missing', 'ambiguous') else 1)

    def test_witness_acceptance_and_rejection(self):
        row = self.missing()
        pred = oracle(row['spec'])
        self.assertTrue(verify(row['spec'], pred))
        pred['worlds'][1] = pred['worlds'][0].copy()
        self.assertFalse(verify(row['spec'], pred))

    def test_rejects_contradiction(self):
        spec = self.missing()['spec']
        pred = oracle(spec)
        pred['worlds'][0]['b'] = 1 - spec['fixed']['b']
        self.assertFalse(verify(spec, pred))

    def test_rejects_malformed_worlds(self):
        spec = self.missing()['spec']
        for value in (None, [], {}, {'a': 0}, {'a': True, 'b': 0}, {'a': 2, 'b': 0},
                      {'a': 0, 'b': 0, 'extra': 1}):
            self.assertFalse(verify(spec, {'worlds': [value, {'a': 1, 'b': 0}]}))

    def test_irrelevant_unknown_is_not_ambiguity(self):
        spec = copy.deepcopy(self.missing()['spec'])
        spec['amounts']['a'] = 0
        self.assertEqual(oracle(spec)['verdict'], 'supported')
        self.assertFalse(verify(spec, {'worlds': list(worlds(spec))}))

    def test_repair_resolves_answer(self):
        row = self.missing()
        spec = copy.deepcopy(row['spec'])
        spec['fixed']['a'] = row['hidden_world']['a']
        self.assertEqual(oracle(spec)['verdict'], 'supported')

    def test_hidden_truth_does_not_constrain_witness(self):
        row = copy.deepcopy(self.missing())
        before = oracle(row['spec'])
        row['hidden_world']['a'] = 1 - row['hidden_world']['a']
        self.assertEqual(before, oracle(row['spec']))

    def test_missing_predictions_count_against_accuracy(self):
        result = evaluate(self.keys, [])
        self.assertEqual(result['metrics']['accuracy_including_abstentions'], 0)
        self.assertEqual(result['counts']['missing'], 80)

    def test_duplicate_and_unknown_predictions_rejected(self):
        p = {'id': self.keys[0]['id'], 'verdict': 'supported'}
        with self.assertRaises(ValueError):
            evaluate(self.keys, [p, p])
        with self.assertRaises(ValueError):
            evaluate(self.keys, [{'id': 'unknown'}])

    def test_known_weakness_v01_is_solved_by_a_regex(self):
        # Pinned on purpose. A text match with no reasoning labels every v0.1 task
        # and emits a valid witness, so v0.1 cannot separate auditors. See README.
        predictions = []
        for task in self.public:
            text = ' '.join(d['text'] for d in task['documents'])
            b = int('Event b is assigned to account T.' in text)
            if re.search(r'Event a is assigned to account [TU]\.', text):
                predictions.append({'id': task['id'], 'verdict': 'supported', 'worlds': []})
            else:
                predictions.append({'id': task['id'], 'verdict': 'underdetermined',
                                    'worlds': [{'a': 0, 'b': b}, {'a': 1, 'b': b}]})
        metrics = evaluate(self.keys, predictions)['metrics']
        self.assertEqual(metrics['accuracy_including_abstentions'], 1)
        self.assertEqual(metrics['verified_detection_recall'], 1)

    def test_symbolic_control(self):
        predictions = [dict(id=k['id'], **oracle(k['spec'])) for k in self.keys]
        result = evaluate(self.keys, predictions)
        self.assertEqual(result['metrics']['accuracy_including_abstentions'], 1)
        self.assertEqual(result['metrics']['verified_detection_recall'], 1)


if __name__ == '__main__':
    unittest.main()
