from pathlib import Path
import pandas as pd
import numpy as np
import json
from .utils import sha256_file, save_json

SCHEMA = {
    "record_id": "text",
    "plot_area_ha": "numeric",
    "rainfall_mm": "numeric",
    "soil_ph": "numeric",
    "seed_kg": "numeric",
    "distance_km": "numeric",
    "arrival_hour": "numeric",
    "actual_yield_kg": "numeric",
    "dispatch_attention": "binary",
}
FEATURE_COLS = ["plot_area_ha", "rainfall_mm", "soil_ph", "seed_kg", "distance_km", "arrival_hour"]
REG_TARGET = "actual_yield_kg"
CLF_TARGET = "dispatch_attention"
ID_COL = "record_id"

def _validate(df):
    expected = list(SCHEMA)
    if set(df.columns) != set(expected) or len(df.columns) != len(expected):
        raise ValueError(f"Columns do not match required schema. Expected {expected}; got {list(df.columns)}")
    if df[ID_COL].isna().any() or df[ID_COL].duplicated().any():
        raise ValueError("record_id must be present and unique")
    for col in FEATURE_COLS + [REG_TARGET]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} must be numeric")
    vals = set(df[CLF_TARGET].dropna().unique().tolist())
    if not vals.issubset({0, 1}):
        raise ValueError("dispatch_attention must contain only 0 and 1 when present")

def load_and_prepare(data_path, output_dir, group_code):
    data_path = Path(data_path); output_dir = Path(output_dir)
    df = pd.read_csv(data_path)
    _validate(df)
    df = df[list(SCHEMA)]
    missing = {c: int(n) for c, n in df.isna().sum().items()}
    duplicate_rows = int(df.duplicated().sum())
    supervised = df.dropna(subset=[REG_TARGET, CLF_TARGET]).copy()
    feature_missing_supervised = {c: int(supervised[c].isna().sum()) for c in FEATURE_COLS}
    X_supervised = supervised[FEATURE_COLS].to_numpy(dtype=float)
    y_reg = supervised[REG_TARGET].to_numpy(dtype=float)
    y_clf = supervised[CLF_TARGET].to_numpy(dtype=int)
    report = {
        "group_code": group_code,
        "row_count": int(len(df)),
        "supervised_row_count": int(len(supervised)),
        "feature_count": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "identifier_column": ID_COL,
        "regression_target": REG_TARGET,
        "classification_target": CLF_TARGET,
        "missing_values": missing,
        "feature_missing_in_supervised_rows": feature_missing_supervised,
        "duplicate_rows": duplicate_rows,
        "descriptive_statistics": json.loads(df.describe(include="all").to_json()),
        "sha256": sha256_file(data_path),
    }
    save_json(report, output_dir/'data_report.json')
    return df, supervised, X_supervised, y_reg, y_clf, FEATURE_COLS
