"""Run from the project root:  python -m unittest discover -s tests -v
Uses a temporary output folder and a temporary models folder copy, so it never touches artifacts/."""
import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'AI_A1_G11.csv'
VALID = {"plot_area_ha": 1.2, "rainfall_mm": 81, "soil_ph": 5.7, "seed_kg": 210, "distance_km": 14, "arrival_hour": 9}


def run(args, cwd):
    return subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True)


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.tmp.name)
        for name in ['run_all.py', 'predict.py']:
            (cls.work / name).write_text((ROOT / name).read_text())
        (cls.work / 'src').mkdir()
        for f in (ROOT / 'src').glob('*.py'):
            (cls.work / 'src' / f.name).write_text(f.read_text())
        (cls.work / 'data').mkdir()
        (cls.work / 'data' / DATA.name).write_bytes(DATA.read_bytes())
        r = run(['run_all.py', '--data', f'data/{DATA.name}', '--output', 'out/', '--group', 'AI-G11'], cls.work)
        cls.rc, cls.stdout, cls.stderr = r.returncode, r.stdout, r.stderr

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_01_pipeline_runs(self):
        self.assertEqual(self.rc, 0, self.stderr)

    def test_02_all_artifacts(self):
        for f in ['data_report.json', 'regression_metrics.json', 'regression_loss.png', 'classification_metrics.json',
                  'confusion_matrix.png', 'clustering_metrics.json', 'clusters.csv', 'cluster_plot.png']:
            self.assertTrue((self.work / 'out' / f).exists(), f)

    def test_03_hash_matches(self):
        import hashlib
        rep = json.loads((self.work / 'out' / 'data_report.json').read_text())
        self.assertEqual(rep['sha256'], hashlib.sha256(DATA.read_bytes()).hexdigest())
        self.assertEqual(rep['row_count'], len(pd.read_csv(DATA)))

    def test_04_regression_metrics_recomputed(self):
        m = json.loads((self.work / 'out' / 'regression_metrics.json').read_text())
        pr = pd.DataFrame(m['test_predictions'])
        mae = float(np.mean(np.abs(pr.prediction - pr.actual)))
        self.assertAlmostEqual(mae, m['mae'], places=6)
        self.assertEqual(m['random_state'], 42)

    def test_05_confusion_matrix_consistent(self):
        m = json.loads((self.work / 'out' / 'classification_metrics.json').read_text())
        (tn, fp), (fn, tp) = m['confusion_matrix']
        self.assertAlmostEqual((tp + tn) / (tp + tn + fp + fn), m['accuracy'], places=9)

    def test_06_every_record_clustered(self):
        c = pd.read_csv(self.work / 'out' / 'clusters.csv')
        self.assertEqual(len(c), len(pd.read_csv(DATA)))
        k = json.loads((self.work / 'out' / 'clustering_metrics.json').read_text())
        self.assertEqual(sorted(k['k_scores']), ['2', '3', '4', '5'])

    def test_07_predict_valid(self):
        r = run(['predict.py', '--record', json.dumps(VALID)], self.work)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        for key in ['regression_prediction', 'classification_prediction', 'classification_probability',
                    'cluster_label', 'group_code', 'model_version']:
            self.assertIn(key, out)
        self.assertTrue(np.isfinite(out['regression_prediction']))

    def test_08_predict_rejects_bad_input(self):
        bad = [{"plot_area_ha": 1.2}, {**VALID, "soil_ph": "x"}, {**VALID, "soil_ph": 99}, {**VALID, "extra": 1}]
        for rec in bad:
            r = run(['predict.py', '--record', json.dumps(rec)], self.work)
            self.assertEqual(r.returncode, 2)
            self.assertIn('error', json.loads(r.stdout))
        r = run(['predict.py', '--record', 'not json'], self.work)
        self.assertEqual(r.returncode, 2)

    def test_09_hidden_like_data(self):
        df = pd.read_csv(DATA)
        df = pd.concat([df] * 5, ignore_index=True)
        df['record_id'] = [f'H{i:03d}' for i in range(len(df))]
        df.loc[2, 'rainfall_mm'] = np.nan
        df[df.columns[::-1]].to_csv(self.work / 'hidden.csv', index=False)
        r = run(['run_all.py', '--data', 'hidden.csv', '--output', 'out_h/', '--group', 'AI-G11'], self.work)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads((self.work / 'out_h' / 'data_report.json').read_text())['row_count'], 100)

    def test_10_live_change_options(self):
        r = run(['run_all.py', '--data', f'data/{DATA.name}', '--output', 'out_l/', '--group', 'AI-G11',
                 '--seed', '7', '--lr', '0.005', '--threshold', '0.3', '--kmin', '2', '--kmax', '4'], self.work)
        self.assertEqual(r.returncode, 0, r.stderr)
        k = json.loads((self.work / 'out_l' / 'clustering_metrics.json').read_text())
        self.assertEqual(sorted(k['k_scores']), ['2', '3', '4'])
        self.assertEqual(k['random_state'], 7)

    def test_11_diverging_lr_gives_clear_error(self):
        r = run(['run_all.py', '--data', f'data/{DATA.name}', '--output', 'out_d/', '--group', 'AI-G11', '--lr', '5'], self.work)
        self.assertEqual(r.returncode, 1)
        self.assertIn('diverged', r.stderr)

    def test_12_gradient_descent_matches_closed_form(self):
        # With enough epochs, batch gradient descent must approach the least-squares solution.
        sys.path.insert(0, str(ROOT))
        from src.regression import train_linear_regression
        df = pd.read_csv(DATA)
        X = df.drop(columns=['record_id', 'actual_yield_kg', 'dispatch_attention']).to_numpy(float)
        y = df['actual_yield_kg'].to_numpy(float)
        out = self.work / 'out_cf'; out.mkdir(exist_ok=True); md = self.work / 'models_cf'; md.mkdir(exist_ok=True)
        m = train_linear_regression(X, y, out, md, random_state=42, learning_rate=0.1, epochs=20000)
        from sklearn.model_selection import train_test_split
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
        w_ls = np.linalg.lstsq(A, ytr, rcond=None)[0]
        self.assertTrue(np.allclose(m['weights'], w_ls, atol=1e-2 * (1 + np.abs(w_ls).max())))


if __name__ == '__main__':
    unittest.main()
