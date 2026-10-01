import argparse, json, sys
from pathlib import Path
import numpy as np, joblib

FIELDS = ["plot_area_ha", "rainfall_mm", "soil_ph", "seed_kg", "distance_km", "arrival_hour"]
# Plausibility ranges (inclusive). Values outside are rejected with a clear message.
RANGES = {
    "plot_area_ha": (0, 1000),
    "rainfall_mm": (0, 5000),
    "soil_ph": (0, 14),
    "seed_kg": (0, 1e6),
    "distance_km": (0, 1000),
    "arrival_hour": (0, 24),
}


def validate(rec):
    if not isinstance(rec, dict):
        raise ValueError('Record must be a JSON object')
    missing = [f for f in FIELDS if f not in rec]
    if missing:
        raise ValueError('Missing field(s): ' + ', '.join(missing))
    extra = set(rec) - set(FIELDS)
    if extra:
        raise ValueError('Unexpected field(s): ' + ', '.join(sorted(extra)))
    for f in FIELDS:
        v = rec[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ValueError(f'Field {f} must be a finite number')
        lo, hi = RANGES[f]
        if not (lo <= v <= hi):
            raise ValueError(f'Field {f} out of plausible range [{lo}, {hi}]: {v}')


def _load_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def main():
    p = argparse.ArgumentParser(description='Predict for one JSON record')
    p.add_argument('--record', required=True)
    p.add_argument('--models-dir', default='models')
    p.add_argument('--group', default=None)
    a = p.parse_args()
    try:
        rec = json.loads(a.record)
        validate(rec)
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        return 2
    md = Path(a.models_dir)
    try:
        reg = np.load(md / 'regression_model.npz')
        clf = joblib.load(md / 'classification_model.joblib')
        clu = joblib.load(md / 'clustering_model.joblib')
    except Exception as e:
        print(json.dumps({'error': 'Models not found. Run run_all.py first.', 'detail': str(e)}))
        return 2
    cfg = _load_json(md / 'run_config.json', {})
    thr = float(_load_json(md / 'classification_config.json', {}).get('decision_threshold', 0.5))
    X = np.array([[rec[f] for f in FIELDS]], dtype=float)
    xs = (X - reg['scaler_mean']) / reg['scaler_std']
    xb = np.c_[np.ones(1), xs]
    reg_pred = float((xb @ reg['weights']).item())
    prob = float(clf.predict_proba(X)[0, 1])
    cp = int(prob >= thr)
    cluster = int(clu['kmeans'].predict(clu['scaler'].transform(clu['imputer'].transform(X)))[0])
    out = {
        'regression_prediction': reg_pred,
        'classification_prediction': cp,
        'classification_probability': prob,
        'cluster_label': cluster,
        'group_code': a.group or cfg.get('group_code', 'UNKNOWN'),
        'model_version': cfg.get('model_version', '1.2'),
    }
    print(json.dumps(out, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
