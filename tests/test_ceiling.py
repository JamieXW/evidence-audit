import itertools
import random
import unittest

from ceiling import (CONTROLS, CORE, STRESS, build, check_certificate, collisions, evaluate,
                     exhaustive, forced_errors, shortcuts, table)


class CeilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks, cls.keys = build()
        cls.by_task = {t['id']: t for t in cls.tasks}

    def pick(self, variant):
        key = next(k for k in self.keys if k['variant'] == variant)
        return self.by_task[key['id']], key

    def run_control(self, name):
        return evaluate(self.tasks, self.keys, [CONTROLS[name](t) for t in self.tasks])

    def test_counts_and_reproducibility(self):
        self.assertEqual((self.tasks, self.keys), build())
        self.assertEqual(len(self.keys), 120)
        self.assertEqual(len({k['scenario'] for k in self.keys}), 20)
        self.assertEqual(len({k['id'] for k in self.keys}), 120)

    def test_family_and_scenario_split(self):
        for field in ('scenario', 'family'):
            dev = {k[field] for k in self.keys if k['split'] == 'dev'}
            test = {k[field] for k in self.keys if k['split'] == 'test'}
            self.assertFalse(dev & test)

    def test_auditor_view_has_no_grading_fields(self):
        allowed = {'id', 'task', 'label', 'agent_sees', 'withheld_from_agent', 'metric',
                   'n_rows', 'start_correct', 'target_correct', 'columns', 'rows'}
        for task in self.tasks:
            self.assertEqual(set(task), allowed)

    def test_labels_follow_reference_solver(self):
        expected = {'complete': 'supported', 'irrelevant': 'supported', 'redundant': 'supported',
                    'withheld': 'unsupported', 'optimistic': 'unsupported', 'sparse': 'unsupported'}
        for key in self.keys:
            target = self.by_task[key['id']]['target_correct']
            self.assertEqual(key['label'], expected[key['variant']])
            self.assertEqual(key['reference_solver_correct'] < target, key['label'] == 'unsupported')

    def test_hard_negatives_have_a_withheld_column(self):
        for key in self.keys:
            if key['variant'] in ('irrelevant', 'redundant'):
                self.assertTrue(self.by_task[key['id']]['withheld_from_agent'])
                self.assertEqual(key['label'], 'supported')

    def test_exact_ceiling_bounds_the_reference_solver(self):
        for key in self.keys:
            self.assertGreaterEqual(key['exact_ceiling_correct'], key['reference_solver_correct'])

    def test_forced_errors_match_brute_force(self):
        rng = random.Random(7)
        for _ in range(50):
            pairs = [(rng.randrange(3), rng.randrange(2)) for _ in range(9)]
            best = max(sum(f[key] == label for key, label in pairs)
                       for f in itertools.product((0, 1), repeat=3))
            self.assertEqual(len(pairs) - forced_errors(pairs), best)

    def test_rejects_bad_groups(self):
        task, _ = self.pick('withheld')
        rows = table(task)
        good = collisions(task)[0]
        same_label = [i for i in rows if rows[i] == rows[good[0]]][:2]
        other_key = next(i for i in rows if rows[i][0] != rows[good[0]][0])
        for bad in (None, [], [good[0]], [good[0], good[0]], [good[0], 9999], [good[0], True],
                    [good[0], str(good[1])], same_label, [good[0], other_key], 'x'):
            cert = check_certificate(task, [bad])
            self.assertEqual((cert['valid_groups'], cert['forced_errors']), (0, 0), bad)
        self.assertEqual(check_certificate(task, 'not a list')['invalid_groups'], 1)

    def test_row_reuse_is_rejected(self):
        task, _ = self.pick('withheld')
        group = collisions(task)[0]
        cert = check_certificate(task, [group, group])
        self.assertEqual((cert['valid_groups'], cert['invalid_groups']), (1, 1))

    def test_partial_certificates_stay_sound(self):
        rng = random.Random(11)
        for key in self.keys:
            task = self.by_task[key['id']]
            groups = collisions(task)
            for _ in range(5):
                subset = [g for g in groups if rng.random() < 0.5]
                bound = check_certificate(task, subset)['ceiling_bound']
                self.assertGreaterEqual(bound, key['exact_ceiling_correct'])

    def test_random_groups_never_refute_a_supported_target(self):
        rng = random.Random(13)
        for key in self.keys:
            if key['label'] != 'supported':
                continue
            task = self.by_task[key['id']]
            for _ in range(20):
                groups = [rng.sample(range(task['n_rows']), rng.randrange(2, 6)) for _ in range(12)]
                self.assertFalse(check_certificate(task, groups)['refutes_target'])

    def test_exhaustive_reference(self):
        result = self.run_control('exhaustive')
        core, stress = result['core']['metrics'], result['stress']['metrics']
        self.assertEqual(core['verdict_accuracy'], 1)
        self.assertEqual(core['certified_recall'], 1)
        self.assertEqual(core['false_certified_refutations'], 0)
        # Known limit: with a near-unique visible column there are no exact collisions.
        self.assertEqual(stress['certified_recall'], 0)

    def test_unsound_auditor_is_caught(self):
        core = self.run_control('sloppy')['core']['metrics']
        self.assertGreater(core['false_alarm_rate'], 0.3)
        self.assertGreater(core['invalid_group_rate'], 0.9)
        self.assertEqual(core['false_certified_refutations'], 0)

    def test_lazy_auditor_is_sound_but_loose(self):
        core = self.run_control('lazy')['core']['metrics']
        self.assertEqual(core['false_certified_refutations'], 0)
        self.assertLess(core['certificate_tightness'], 0.5)

    def test_surface_shortcuts_stay_near_majority(self):
        result = shortcuts(self.tasks, self.keys)
        majority = result.pop('majority_rate')
        result.pop('n_core')
        for name, accuracy in result.items():
            self.assertLessEqual(accuracy, majority + 0.10, name)

    def test_missing_and_malformed_predictions(self):
        result = evaluate(self.tasks, self.keys, [])
        self.assertEqual(result['core']['metrics']['verdict_accuracy'], 0)
        self.assertEqual(result['core']['counts']['total'], 20 * len(CORE))
        self.assertEqual(result['stress']['counts']['total'], 20 * len(STRESS))
        p = exhaustive(self.tasks[0])
        with self.assertRaises(ValueError):
            evaluate(self.tasks, self.keys, [p, p])
        with self.assertRaises(ValueError):
            evaluate(self.tasks, self.keys, [{'id': 'unknown'}])


if __name__ == '__main__':
    unittest.main()
