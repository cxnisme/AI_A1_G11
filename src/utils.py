import hashlib, json
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
