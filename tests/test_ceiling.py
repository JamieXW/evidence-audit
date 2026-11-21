import copy
import itertools
import random
import unittest

from fractions import Fraction

from ceiling import (CONTROLS, CORE, N_PRACTICE, STRESS, best_pairwise_f1, build,
                     build_practice, check_certificate, check_f1_certificate, choose_solver,
                     collisions, evaluate, exhaustive, forced_errors, interval,
                     interval_report, shortcuts, signal_check, table, toy_clonal_task)


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

    def test_bad_column_declarations_void_the_certificate(self):
        task, _ = self.pick('withheld')
        groups = collisions(task)
        sees, hidden = task['agent_sees'], task['withheld_from_agent'][0]
        for bad in ([], 'income_band', [sees[0], sees[0]], sees + [hidden], [hidden], [1], ['nope']):
            cert = check_certificate(task, groups, bad)
            self.assertEqual((cert['scope'], cert['forced_errors'], cert['refutes_target']),
                             (None, 0, False), bad)

    def test_scope_follows_declared_columns(self):
        task, _ = self.pick('withheld')
        sees = task['agent_sees']
        self.assertEqual(check_certificate(task, collisions(task))['scope'], 'any_solver')
        self.assertEqual(check_certificate(task, collisions(task), sees[::-1])['scope'], 'any_solver')
        narrow = check_certificate(task, collisions(task, sees[:2]), sees[:2])
        self.assertEqual(narrow['scope'], 'restricted')
        # Groups matched on fewer columns are not evidence about every solver.
        self.assertEqual(check_certificate(task, collisions(task, sees[:2]))['valid_groups'], 0)

    def test_restricted_bound_on_sparse_equals_withheld_ceiling(self):
        by_scenario = {(k['scenario'], k['variant']): k for k in self.keys}
        for (scenario, variant), key in by_scenario.items():
            if variant != 'sparse':
                continue
            task = self.by_task[key['id']]
            bands = task['agent_sees'][:3]
            cert = check_certificate(task, collisions(task, bands), bands)
            self.assertEqual(cert['scope'], 'restricted')
            self.assertEqual(cert['ceiling_bound'],
                             by_scenario[(scenario, 'withheld')]['exact_ceiling_correct'])

    def test_informed_subset_recovers_the_stress_set(self):
        result = self.run_control('subset_near_unique')
        core, stress = result['core']['metrics'], result['stress']['metrics']
        self.assertEqual(core['certified_recall'], 1)
        self.assertEqual(core['restricted_refutations_on_supported'], 0)
        self.assertEqual(stress['certified_recall'], 0)
        self.assertEqual(stress['restricted_recall'], 1)
        self.assertEqual(stress['restrictions_that_cut_signal'], 0)

    def test_careless_subset_misleads_but_is_never_certified_for_all_solvers(self):
        core = self.run_control('subset_careless')['core']['metrics']
        self.assertEqual(core['false_certified_refutations'], 0)
        self.assertEqual(core['certified_recall'], 0)
        self.assertGreater(core['restricted_refutations_on_supported'], 0)
        self.assertGreater(core['restrictions_that_cut_signal'], 0)

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


class SignalCheckTests(unittest.TestCase):
    """Task: does a restricted certificate's dropped column carry signal?"""

    @classmethod
    def setUpClass(cls):
        cls.tasks, cls.keys = build()
        cls.by_task = {t['id']: t for t in cls.tasks}
        cls.practice = build_practice()

        def scored(name):
            preds = [CONTROLS[name](t) for t in cls.tasks]
            return evaluate(cls.tasks, cls.keys, preds, cls.practice)
        cls.careless = scored('subset_careless')
        cls.near_unique = scored('subset_near_unique')

    def test_practice_matches_tasks_and_is_reproducible(self):
        self.assertEqual(self.practice, build_practice())
        self.assertEqual(set(self.practice), set(self.by_task))
        for task_id, prac in self.practice.items():
            self.assertEqual(prac['columns'], self.by_task[task_id]['columns'])
            self.assertEqual(len(prac['rows']), N_PRACTICE)

    def test_practice_rows_differ_from_evaluation_rows(self):
        for task_id, prac in self.practice.items():
            hidden = {tuple(r[1:]) for r in self.by_task[task_id]['rows']}
            practice = [tuple(r[1:]) for r in prac['rows']]
            # Coarse features repeat, so overlap is expected; the tables must not be copies.
            self.assertLess(sum(r in hidden for r in practice), N_PRACTICE)

    def test_check_needs_a_proper_subset(self):
        task = self.tasks[0]
        prac = self.practice[task['id']]
        sees = task['agent_sees']
        for keep in (None, [], sees, list(reversed(sees)), sees + ['nope'], 'x'):
            self.assertIsNone(signal_check(task, prac, keep), keep)

    def test_check_is_deterministic(self):
        task = next(t for t in self.tasks if t['withheld_from_agent'])
        prac = self.practice[task['id']]
        keep = task['agent_sees'][:2]
        self.assertEqual(signal_check(task, prac, keep), signal_check(task, prac, keep))

    def test_flags_careless_misleading_refutations(self):
        core = self.careless['core']['metrics']
        flagged, total = map(int, core['misleading_restricted_refutations_flagged'].split(' of '))
        self.assertEqual(total, 28)
        self.assertGreaterEqual(flagged, 25)
        self.assertEqual(core['promoted_refutations_on_supported'], 0)
        self.assertLessEqual(core['signal_flags_where_not_cut'], 3)

    def test_passes_near_unique_subsets(self):
        stress = self.near_unique['stress']['metrics']
        self.assertEqual(stress['restrictions_checked'], 20)
        self.assertEqual(stress['signal_flags_where_not_cut'], 0)
        self.assertEqual(stress['promoted_recall'], 1)

    def test_promoted_results_are_never_certified_for_every_solver(self):
        for result in (self.careless, self.near_unique):
            for part in ('core', 'stress'):
                self.assertEqual(result[part]['metrics']['false_certified_refutations'], 0)
        self.assertEqual(self.near_unique['stress']['metrics']['certified_recall'], 0)

    def test_noise_column_is_rarely_flagged(self):
        flags = checked = 0
        for key in self.keys:
            if key['variant'] not in CORE:
                continue
            task = copy.deepcopy(self.by_task[key['id']])
            prac = copy.deepcopy(self.practice[key['id']])
            rng = random.Random('noise/' + key['id'])
            for tab in (task, prac):
                tab['columns'] = tab['columns'][:1] + ['noise'] + tab['columns'][1:]
                tab['rows'] = [r[:1] + [rng.randrange(2)] + r[1:] for r in tab['rows']]
            keep = list(task['agent_sees'])
            task['agent_sees'] = ['noise'] + keep
            flags += signal_check(task, prac, keep)['signal_detected']
            checked += 1
        self.assertEqual(checked, 100)
        self.assertLessEqual(flags, 10)


def f1(tp, fp, positives):
    return Fraction(2 * tp, tp + fp + positives)


class PairwiseF1Tests(unittest.TestCase):
    """Task: a ceiling for pairwise F1, as in the post's clonal-grouping case."""

    @classmethod
    def setUpClass(cls):
        cls.toy = toy_clonal_task()
        cls.local = ['same_gene', 'mismatches']

    def test_prefix_matches_brute_force(self):
        rng = random.Random(5)
        for _ in range(300):
            groups = [(rng.randrange(1, 9), rng.randrange(1, 9)) for _ in range(rng.randrange(1, 8))]
            free = rng.randrange(0, 6)   # positives outside every group
            total = free + sum(p for p, _ in groups)
            best = max(f1(free + sum(g[0] for g in pick), sum(g[1] for g in pick), total)
                       for r in range(len(groups) + 1)
                       for pick in itertools.combinations(groups, r))
            self.assertEqual(best_pairwise_f1(groups, total, free)[0], best, groups)

    def test_posts_forty_pair_example(self):
        # 40 indistinguishable pairs, 30 truly the same: calling them all "same" is best.
        best, included = best_pairwise_f1([(30, 10)], 30, 0)
        self.assertEqual((best, included), (Fraction(6, 7), 1))

    def test_no_positive_pairs(self):
        self.assertEqual(best_pairwise_f1([], 0, 0), (None, 0))

    def test_bound_is_exact_for_pair_rules(self):
        # Every rule mapping a feature tuple to same/different, checked one by one.
        for columns in (self.local, self.toy['agent_sees']):
            rows = table(self.toy, columns).values()
            keys = sorted({k for k, _ in rows})
            positives = sum(y for _, y in rows)
            best = Fraction(0)
            for calls in itertools.product((0, 1), repeat=len(keys)):
                rule = dict(zip(keys, calls))
                tp = sum(rule[k] and y for k, y in rows)
                fp = sum(rule[k] and not y for k, y in rows)
                best = max(best, f1(tp, fp, positives))
            cert = check_f1_certificate(self.toy, collisions(self.toy, columns), columns)
            self.assertEqual(cert['f1_ceiling'], round(float(best), 4))

    def test_partial_certificates_stay_sound(self):
        groups = collisions(self.toy)
        exact = check_f1_certificate(self.toy, groups)['f1_ceiling']
        rng = random.Random(3)
        for _ in range(50):
            subset = [g for g in groups if rng.random() < 0.5]
            self.assertGreaterEqual(check_f1_certificate(self.toy, subset)['f1_ceiling'], exact)

    def test_clustering_escapes_a_pair_feature_bound(self):
        # Linking through other cells is information the two pair features lack.
        cert = check_f1_certificate(self.toy, collisions(self.toy, self.local), self.local)
        self.assertEqual(cert['scope'], 'restricted')
        self.assertGreater(self.toy['start_f1'], cert['f1_ceiling'])

    def test_chain_feature_brings_the_start_inside_the_bound(self):
        cert = check_f1_certificate(self.toy, collisions(self.toy))
        self.assertEqual(cert['scope'], 'pair_rules')
        self.assertGreaterEqual(cert['f1_ceiling'], self.toy['start_f1'])
        self.assertTrue(cert['refutes_target'])

    def test_bad_input(self):
        groups = collisions(self.toy)
        rows = table(self.toy)
        same_label = [i for i in rows if rows[i] == rows[groups[0][0]]][:2]
        for bad in ([groups[0][0]], [groups[0][0], groups[0][0]], same_label, [10 ** 6, 1]):
            self.assertEqual(check_f1_certificate(self.toy, [bad])['valid_groups'], 0, bad)
        self.assertIsNone(check_f1_certificate(self.toy, groups, ['row'])['scope'])
        broken = copy.deepcopy(self.toy)
        broken['rows'][0][-1] = 2
        with self.assertRaises(ValueError):
            check_f1_certificate(broken, groups)


class IntervalTests(unittest.TestCase):
    """Task: floor from a concrete solver, certified ceiling, cross-fit estimate."""

    @classmethod
    def setUpClass(cls):
        cls.tasks, cls.keys = build()
        cls.by_task = {t['id']: t for t in cls.tasks}
        cls.practice = build_practice()
        cls.report = interval_report(cls.tasks, cls.keys, cls.practice)

    def test_floor_never_exceeds_ceiling(self):
        for d in self.report['details']:
            self.assertLessEqual(d['floor'], d['ceiling'], d['id'])

    def test_ceiling_matches_exact_ceiling(self):
        exact = {k['id']: k['exact_ceiling_correct'] for k in self.keys}
        for d in self.report['details']:
            self.assertEqual(d['ceiling'], exact[d['id']])

    def test_solver_choice_ignores_evaluation_labels(self):
        for task in self.tasks[:30]:
            prac = self.practice[task['id']]
            flipped = copy.deepcopy(task)
            for row in flipped['rows']:
                row[-1] = 1 - row[-1]
            self.assertEqual(interval(task, prac)['floor_solver_columns'],
                             interval(flipped, prac)['floor_solver_columns'])
            self.assertEqual(choose_solver(task, prac['rows']), choose_solver(flipped, prac['rows']))

    def test_no_status_contradicts_the_generator(self):
        counts = self.report['by_truth_and_status']
        self.assertNotIn('unsupported/witnessed', counts)
        self.assertNotIn('supported/refuted', counts)

    def test_core_tasks_are_almost_all_resolved(self):
        core = [d for d in self.report['details'] if d['variant'] in CORE]
        self.assertEqual(len(core), 100)
        self.assertGreaterEqual(sum(d['status'] != 'open' for d in core), 95)

    def test_sparse_stays_open_without_a_restriction(self):
        sparse = [d for d in self.report['details'] if d['variant'] == 'sparse']
        self.assertEqual({d['status'] for d in sparse}, {'open'})
        self.assertEqual({d['ceiling'] for d in sparse}, {72})

    def test_report_is_deterministic(self):
        self.assertEqual(self.report, interval_report(self.tasks, self.keys, self.practice))


if __name__ == '__main__':
    unittest.main()
