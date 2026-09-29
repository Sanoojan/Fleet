import unittest
import numpy as np
from fleet.train.continual_metrics import (
    detection_metrics, fake_probability, final_summary, step_summary, validate_threshold,
)


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.r = np.array([[.5, .4, .3], [.8, .5, .4], [.7, .9, .6], [.6, .8, .85]])

    def test_indexing(self):
        s = step_summary(self.r)
        self.assertAlmostEqual(s['current_adaptation_gain'], .25)
        self.assertAlmostEqual(s['running_bwt'], -.15)
        self.assertAlmostEqual(s['running_forgetting'], .15)
        f = final_summary(self.r, ['a', 'b', 'c'])
        for key, value in dict(bwt=-.15, fwt=.2, forgetting=.15,
                               forward_performance=1.6/3, mean_adaptation_gain=.95/3,
                               average_incremental=(.8 + .8 + 2.25/3)/3).items():
            self.assertAlmostEqual(f[key], value)
        self.assertEqual(f['previous_declined_count'], 2)
        self.assertIsNone(step_summary(self.r[:1])['seen_macro'])
        self.assertIsNone(step_summary(self.r[:2])['running_bwt'])

    def test_running_and_final_forgetting_differ(self):
        r = [[.4, .4], [.5, .4], [.8, .7]]
        self.assertAlmostEqual(step_summary(r)['running_forgetting'], -.3)
        self.assertEqual(final_summary(r, ['a', 'b'])['forgetting'], 0)

    def test_fixed_threshold_and_tie(self):
        for value in [.49, .51, 0, 1, True, '0.5']:
            with self.assertRaises(ValueError):
                validate_threshold(value)
        validate_threshold(.5)
        confidence = np.array([-.8, -1e-20, 0., 1e-20, .8])
        np.testing.assert_array_equal(fake_probability(confidence) >= .5, confidence <= 0)
        m = detection_metrics([1, 0, 0, 1], [.5, .5, .49, .49])
        self.assertEqual([m[k] for k in ['tp', 'tn', 'fp', 'fn']], [1, 1, 1, 1])
        self.assertEqual(m['balanced_accuracy'], .5)
        with self.assertRaises(ValueError):
            detection_metrics([0, 1], [.2, .8], .6)

    def test_config_rejects_changed_threshold(self):
        import tempfile
        from pathlib import Path
        import yaml
        from fleet.train.treasure_continual_data import ROOT, load_config
        original = ROOT / 'Configs/treasure_continual_fleet_5shot_aigibench_replay.yaml'
        c = yaml.safe_load(original.read_text())
        c['decision_threshold'] = .51
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.yaml'
            path.write_text(yaml.safe_dump(c))
            with self.assertRaisesRegex(ValueError, 'exactly 0.5'):
                load_config(path)

    def test_reject_incomplete_final(self):
        with self.assertRaises(ValueError):
            final_summary(self.r[:3], ['a', 'b', 'c'])


class ManifestTest(unittest.TestCase):
    def test_support_exclusion_and_determinism(self):
        import tempfile
        from pathlib import Path
        from fleet.train.treasure_continual_data import exclude_support, sample, stream_seed
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            for i in range(12):
                p = root / f'{i}.png'
                p.write_text(str(i))
                files.append(str(p))
            a = sample(files, 5, stream_seed(42, 'model'))
            self.assertEqual(a, sample(list(reversed(files)), 5, stream_seed(42, 'model')))
            alias = root / 'alias.png'
            alias.symlink_to(a[0])
            hardlink = root / 'hardlink.png'
            hardlink.hardlink_to(a[1])
            query = exclude_support(files + [str(alias), str(hardlink)], a)
            self.assertEqual(len(query), 7)
            self.assertFalse(set(a) & set(query))

    def test_export_65_by_64_and_pre_post(self):
        import csv
        import tempfile
        from pathlib import Path
        from fleet.train.treasure_continual_5shot import export_metrics
        metric = detection_metrics([0, 1], [.1, .9])
        names = [f'generator{i}' for i in range(64)]
        records = [dict(state=i, adapted_generator=names[i-1] if i else '',
                        checkpoint_identity=f'state{i}', elapsed_seconds=i,
                        domains=[metric]*64, retention=metric, training=[], runtime={}, replay={}) for i in range(65)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            matrices = export_metrics({'seed': 42}, records, names, out)
            self.assertTrue(all(m.shape == (65, 64) for m in matrices.values()))
            rows = list(csv.DictReader((out / 'metrics/metrics_long.csv').open()))
            self.assertEqual(len(rows), 65*64)
            self.assertEqual(sum(r['current_generator_phase'] == 'pre' for r in rows), 64)
            self.assertEqual(sum(r['current_generator_phase'] == 'post' for r in rows), 64)
            self.assertEqual(rows[0]['decision_threshold'], '0.5')
            final = rows[-64:]
            self.assertEqual(sum(r['relation'] == 'future' for r in final), 0)


class ResumeTest(unittest.TestCase):
    def test_completed_state_resume_and_changed_config_rejection(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        from fleet.train import treasure_continual_5shot as runner
        from fleet.train.treasure_continual_data import load_config

        c = load_config('Configs/treasure_continual_fleet_5shot_aigibench_replay.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save({'model_state_dict': {}}, root / 'pretrained.pth')
            for cls in ('fake', 'real'):
                np.save(root / f'{cls}_prototype.npy', np.ones(2, dtype=np.float32))
            (root / 'order.csv').write_text('test fixture')
            c.update(output_dir=str(root / 'output'), pretrained_checkpoint=str(root / 'pretrained.pth'),
                     prototype_path=str(root), generator_order_file=str(root / 'order.csv'),
                     max_generators=2, adaptation_epochs=1)
            order = [{'directory': 'a'}, {'directory': 'b'}]
            data = dict(supports={n: {'fake': [], 'real': []} for n in ('a', 'b')})
            evaluations, updates = [], []

            def fake_adapt(c, model, *args):
                state = args[-1]
                updates.append(state)
                with torch.no_grad():
                    model.weight.add_(1)
                return {'fixture': state}, []

            def fake_evaluate(c, model, *args):
                evaluations.append(float(model.weight.item()))
                m = detection_metrics([0, 1], [.2, .8])
                return [dict(m), dict(m)], dict(m)

            def fake_replay(c, data, state):
                return dict(paths=[], labels=[], fake_count=0, real_count=0, seed=state)

            def toy_model(*args):
                model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    model.weight.zero_()
                return model

            with patch.object(runner, 'create_model', toy_model), \
                 patch.object(runner, 'adapt', fake_adapt), \
                 patch.object(runner, 'evaluate', fake_evaluate), \
                 patch.object(runner, 'replay_manifest', fake_replay), \
                 patch.object(runner, 'plots'), \
                 patch('transformers.AutoImageProcessor.from_pretrained', return_value=None), \
                 patch('torch.cuda.is_available', return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run(dict(c), order, data)
                self.assertEqual(evaluations, [0., 1., 2.])
                self.assertEqual(updates, [1, 2])
                runner.run(dict(c), order, data)
                self.assertEqual(len(evaluations), 3)  # Complete resume performs no extra evaluation.
                with self.assertRaisesRegex(ValueError, 'Resume configuration'):
                    runner.run(dict(c, seed=43), order, data)
                # Simulate restart from the committed pretrained state, retaining derived files.
                import shutil
                shutil.copyfile(root / 'output/checkpoints/state_000.pth', root / 'output/checkpoints/latest.pth')
                runner.run(dict(c), order, data)
                self.assertEqual(evaluations, [0., 1., 2., 1., 2.])
                self.assertEqual(updates, [1, 2, 1, 2])
                self.assertTrue((root / 'output/checkpoints/state_000.pth').exists())
                self.assertTrue((root / 'output/checkpoints/best.pth').exists())
                self.assertFalse((root / 'output/checkpoints/state_001.pth').exists())


class LauncherTest(unittest.TestCase):
    def test_duplicate_guard_force_and_quoted_paths(self):
        import os
        import subprocess
        import tempfile
        import time
        from pathlib import Path
        from fleet.train.treasure_continual_data import ROOT
        with tempfile.TemporaryDirectory(dir=ROOT / 'All_outputs', prefix='.launcher_test_') as tmp:
            root = Path(tmp)
            output = root / 'output with spaces'
            output.mkdir()
            stub = root / 'python with spaces'
            stub.write_text("#!/usr/bin/env python3\nimport os,sys\n"
                            "if '--print-output-dir' in sys.argv: print(os.environ['FLEET_TEST_OUTPUT'])\n"
                            "elif '--validate-config' not in sys.argv: print(os.environ['CUDA_VISIBLE_DEVICES'], os.environ['PYTHONPATH'])\n")
            stub.chmod(0o755)
            (output / 'experiment.pid').write_text(str(os.getpid()))
            env = dict(os.environ, PYTHON=str(stub), CUDA_VISIBLE_DEVICES='7',
                       FLEET_TEST_OUTPUT=str(output), FORCE='0')
            cmd = ['bash', str(ROOT / 'run_treasure_continual_5shot_background.sh'), 'config with spaces.yaml']
            blocked = subprocess.run(cmd, cwd='/tmp', env=env, text=True, capture_output=True)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn('refusing duplicate', blocked.stderr)
            launched = subprocess.run(cmd, cwd='/tmp', env=dict(env, FORCE='1'), text=True, capture_output=True)
            self.assertEqual(launched.returncode, 0, launched.stderr)
            self.assertIn('PID:', launched.stdout)
            log = next((output / 'logs').glob('continual_*.log'))
            for _ in range(100):
                if log.stat().st_size:
                    break
                time.sleep(.01)
            self.assertIn('7 ', log.read_text())
            self.assertIn(str(ROOT / 'src'), log.read_text())


class TrackingTest(unittest.TestCase):
    def test_unavailable_tracking_keeps_local_work_possible(self):
        from unittest.mock import patch
        from fleet.train.treasure_continual_5shot import Tracking
        from fleet.train.treasure_continual_data import load_config
        c = load_config('Configs/treasure_continual_fleet_5shot_aigibench_replay.yaml')
        c['wandb']['mode'] = 'offline'
        c['run_id'] = 'test'
        import types
        from unittest.mock import Mock
        fake_wandb = types.SimpleNamespace(init=Mock(side_effect=RuntimeError('fixture unavailable')))
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {'wandb': fake_wandb}):
            c['output_dir'] = tmp
            with self.assertWarnsRegex(UserWarning, 'W&B unavailable'):
                tracker = Tracking(c)
        self.assertIsNone(tracker.run)
        tracker.log({'continual/step': 0})


if __name__ == '__main__':
    unittest.main()
