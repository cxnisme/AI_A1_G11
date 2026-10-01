import argparse
from pathlib import Path
import json
import numpy as np
from src.data_pipeline import load_and_prepare
from src.regression import train_linear_regression
from src.classification import train_classifier
from src.clustering import train_clustering
from src.utils import ensure_dir

def main():
    p=argparse.ArgumentParser(description='Musanze HarvestLink AI pipeline')
    p.add_argument('--data',required=True); p.add_argument('--output',required=True); p.add_argument('--group',required=True); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=0.01); p.add_argument('--epochs',type=int,default=5000); p.add_argument('--threshold',type=float,default=0.5); p.add_argument('--kmin',type=int,default=2); p.add_argument('--kmax',type=int,default=5)
    a=p.parse_args(); out=Path(a.output); ensure_dir(out); ensure_dir('models')
    df,clean,X,yreg,yclf,features=load_and_prepare(a.data,out,a.group)
    reg=train_linear_regression(X,yreg,out,Path('models'),random_state=a.seed,learning_rate=a.lr,epochs=a.epochs)
    train_classifier(X,yclf,features,out,Path('models'),random_state=a.seed,threshold=a.threshold)
    train_clustering(df[features].to_numpy(dtype=float),df['record_id'].tolist(),features,out,Path('models'),random_state=a.seed,k_min=a.kmin,k_max=a.kmax)
    (Path('models')/'run_config.json').write_text(json.dumps({'group_code':a.group,'seed':a.seed,'model_version':'1.2'},indent=2))
    print(f'Group: {a.group}')
    report=json.loads((out/'data_report.json').read_text())
    print(f'Rows: {report["row_count"]}')
    print(f'Features: {report["feature_count"]}')
    print(f'SHA-256: {report["sha256"]}')
    print('Artifacts generated:', ', '.join(sorted(x.name for x in out.iterdir())))

if __name__=='__main__':
    import sys
    try: main()
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr); sys.exit(1)
