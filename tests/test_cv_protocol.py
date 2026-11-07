"""Patient split/model-selection tests. No study data or GPU is used."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'align_utils'))
from cv_protocol import (BestCheckpoint, make_inner_split, record_fold_split,
                         save_json, validate_three_way_split)


class ProtocolTests(unittest.TestCase):
    def test_three_way_partition_rejects_overlap_missing_and_duplicates(self):
        validate_three_way_split([0, 1], [2], [3], 4)
        for groups in (([0, 1], [1, 2], [3]), ([0], [1], [2]),
                       ([0, 0], [1, 2], [3]), ([], [0, 1], [2, 3])):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                validate_three_way_split(*groups, 4)

    def test_validation_alone_selects_checkpoint_without_stopping(self):
        selection = BestCheckpoint()
        for epoch, score in enumerate([0.6, 0.8] + [0.7] * 148, 1):
            selection.update(score, epoch)
        self.assertEqual(selection.best_epoch, 2)
        self.assertEqual(selection.best_score, 0.8)
        self.assertFalse(hasattr(selection, 'patience'))
        self.assertFalse(hasattr(selection, 'should_stop'))
        with self.assertRaises(ValueError):
            selection.update(float('nan'), 151)

    def test_ties_preserve_task_selection_policy(self):
        first, last = BestCheckpoint(), BestCheckpoint(allow_ties=True)
        for epoch in (1, 2, 3):
            first.update(0.7, epoch)
            last.update(0.7, epoch)
        self.assertEqual(first.best_epoch, 1)
        self.assertEqual(last.best_epoch, 3)

    def test_manifests_preserve_patient_identity_and_reject_changed_splits(self):
        with tempfile.TemporaryDirectory(prefix='.cv-test-', dir=ROOT) as directory:
            path = Path(directory) / 'split.json'
            args = (path, ['a', 'b', 'c', 'd'], [0, 1], [2], [3], 42, 0, 1051, 0.2, 'ACC', 100)
            value = record_fold_split(*args)
            record_fold_split(*args)
            self.assertEqual(value['test_patients'], ['d'])
            self.assertFalse(value['early_stopping'])
            with self.assertRaises(ValueError):
                record_fold_split(path, ['a', 'b', 'c', 'd'], [0, 2], [1], [3], 42, 0, 1051, .2, 'ACC', 100)
            with self.assertRaises(ValueError):
                record_fold_split(path, ['a', 'a', 'c', 'd'], [0, 1], [2], [3], 42, 0, 1051, .2, 'ACC', 100)

    def test_json_records_undefined_metric_as_null(self):
        with tempfile.TemporaryDirectory(prefix='.cv-test-', dir=ROOT) as directory:
            path = Path(directory) / 'result.json'
            save_json(path, {'metric': float('nan'), 'folds': [0.7, float('inf')]})
            self.assertEqual(json.loads(path.read_text()), {'metric': None, 'folds': [0.7, None]})

    def test_production_epoch_loops_have_no_break_or_test_indices(self):
        for filename, epochs, folds in (('6-Classification.py', 100, 6), ('7-Regression.py', 150, 10)):
            source = ROOT / 'prognostic_prediction' / filename
            tree = ast.parse(source.read_text())
            values = {node.targets[0].id: ast.literal_eval(node.value)
                      for node in tree.body if isinstance(node, ast.Assign)
                      and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id in ('MAX_EPOCHS', 'N_SPLITS')}
            self.assertEqual(values, {'MAX_EPOCHS': epochs, 'N_SPLITS': folds})
            loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                     and isinstance(node.target, ast.Name) and node.target.id == 'epoch']
            self.assertEqual(len(loops), 1)
            names = {node.id for node in ast.walk(loops[0]) if isinstance(node, ast.Name)}
            self.assertNotIn('test_idx', names)
            self.assertNotIn('test_loader', names)
            self.assertNotIn('test_df', names)
            self.assertFalse(any(isinstance(node, (ast.Break, ast.Return)) for node in ast.walk(loops[0])))

    def test_actual_stratified_splits_are_deterministic_and_complete(self):
        from sklearn.model_selection import StratifiedKFold
        labels = [i % 2 for i in range(83)]
        for folds in (6, 10):
            for seed in (42, 44, 46, 48, 50, 52):
                counts = [0] * len(labels)
                for fold, (development, test) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(labels, labels)):
                    first = make_inner_split(development, test, labels, seed, fold)
                    second = make_inner_split(development, test, labels, seed, fold)
                    self.assertEqual(first, second)
                    train, validation, _ = first
                    validate_three_way_split(train, validation, test, len(labels))
                    self.assertEqual({labels[i] for i in train}, {0, 1})
                    self.assertEqual({labels[i] for i in validation}, {0, 1})
                    altered_test_labels = labels.copy()
                    for i in test:
                        altered_test_labels[i] = 1 - altered_test_labels[i]
                        counts[i] += 1
                    self.assertEqual(first, make_inner_split(development, test, altered_test_labels, seed, fold))
                self.assertEqual(counts, [1] * len(labels))


class SyntheticIntegrationTests(unittest.TestCase):
    """Run reduced CPU-only toy folds, never any real experiment."""
    def test_actual_train_loops_select_internally_and_read_test_only_after_selection(self):
        import torch
        import pandas as pd
        torch.set_num_threads(1)
        for filename, function_name, model_name, output_name in (
            ('6-Classification.py', 'train_and_validate', 'SelfCalibrationMIL', 'CHECKPOINT_ROOT'),
            ('7-Regression.py', 'run_experiment', 'GatedMTLSurvivalMIL', 'SAVE_DIR'),
        ):
            with self.subTest(task=filename), tempfile.TemporaryDirectory(prefix='.cv-test-', dir=ROOT) as directory:
                spec = importlib.util.spec_from_file_location('synthetic_' + filename[0], ROOT / 'prognostic_prediction' / filename)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.DEVICE = torch.device('cpu')
                module.N_SPLITS = 2
                module.MAX_EPOCHS = 3  # Tests only; production constants are checked separately.
                setattr(module, output_name, Path(directory) / 'checkpoints')
                features = Path(directory) / 'features'
                features.mkdir()
                ids = [f'toy_{i:02d}' for i in range(24)]
                data = pd.DataFrame({'case_submitter_id': ids, 'label': [i % 2 for i in range(24)],
                                     'OS_days': [float(i + 1) for i in range(24)], 'event_status': [1] * 24})
                for patient in ids:
                    torch.save({'features': torch.randn(4, 64)}, features / f'{patient}.pt')
                original_load, original_model = torch.load, getattr(module, model_name)
                events = []

                def make_model(*args, **kwargs):
                    events.append(('start_fold', None))
                    return original_model(*args, **kwargs)

                def tracked_load(path, *args, **kwargs):
                    events.append(('load', Path(path).name))
                    return original_load(path, *args, **kwargs)

                # Make internal validation plateau. Fixed-length training must still
                # reach every epoch, and test scoring must happen after reloading.
                if filename.startswith('6-'):
                    metric_patch = mock.patch.object(module, 'accuracy_score', return_value=0.5)
                else:
                    metric_patch = mock.patch.object(module, 'concordance_index', return_value=0.5)
                with mock.patch.object(module, model_name, side_effect=make_model), \
                     mock.patch.object(torch, 'load', side_effect=tracked_load), metric_patch:
                    result = getattr(module, function_name)(42, data, features)
                self.assertEqual(result['coverage'], len(ids))
                self.assertEqual(len(result['folds']), 2)
                starts = [i for i, event in enumerate(events) if event[0] == 'start_fold'] + [len(events)]
                self.assertEqual(len(starts), 3)
                seen_test = []
                for fold, record in enumerate(result['folds']):
                    self.assertEqual(record['epochs_run'], 3)
                    self.assertEqual(len(record['history']), 3)
                    self.assertEqual(record['best_epoch'], 3 if filename.startswith('6-') else 1)
                    validate_three_way_split(
                        [ids.index(p) for p in record['train_patients']],
                        [ids.index(p) for p in record['validation_patients']],
                        [ids.index(p) for p in record['test_patients']], len(ids))
                    activity = events[starts[fold]:starts[fold + 1]]
                    reloads = [i for i, event in enumerate(activity) if event[0] == 'load' and event[1].endswith('.pth')]
                    self.assertEqual(len(reloads), 1)
                    before = [name for kind, name in activity[:reloads[0]] if kind == 'load']
                    after = [name for kind, name in activity[reloads[0] + 1:] if kind == 'load']
                    test_files = [patient + '.pt' for patient in record['test_patients']]
                    self.assertFalse(set(before) & set(test_files))
                    self.assertEqual(after, test_files)
                    seen_test.extend(record['test_patients'])
                self.assertEqual(sorted(seen_test), sorted(ids))


if __name__ == '__main__':
    unittest.main()
