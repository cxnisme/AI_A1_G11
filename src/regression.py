from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from .utils import save_json

def train_linear_regression(X, y, output_dir, model_dir, random_state=42, test_size=0.2, learning_rate=0.01, epochs=5000):
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(y)), test_size=test_size, random_state=random_state
    )
    medians = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)
    mean = X_train.mean(axis=0); std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    Xtr = (X_train-mean)/std; Xte=(X_test-mean)/std
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xte=np.c_[np.ones(len(Xte)), Xte]
    w=np.zeros(Xtr.shape[1], dtype=float); loss=[]
    for _ in range(epochs):
        pred=Xtr@w; err=pred-y_train; cur=float(np.mean(err**2))
        if not np.isfinite(cur) or cur>1e30:
            raise ValueError(f'Gradient descent diverged at epoch {len(loss)} (loss is not finite). Use a smaller --lr (current: {learning_rate}).')
        loss.append(cur)
        grad=(2.0/len(y_train))*(Xtr.T@err); w -= learning_rate*grad
    pred_test=Xte@w
    mae=float(np.mean(np.abs(pred_test-y_test)))
    rmse=float(np.sqrt(np.mean((pred_test-y_test)**2)))
    ss_res=float(np.sum((y_test-pred_test)**2)); ss_tot=float(np.sum((y_test-y_test.mean())**2))
    r2=float(1-ss_res/ss_tot) if ss_tot>0 else 0.0
    pred_rows=[{"row_index":int(i),"actual":float(a),"prediction":float(p)} for i,a,p in zip(test_idx,y_test,pred_test)]
    metrics={"model_type":"linear_regression_from_first_principles","random_state":random_state,"test_size":test_size,"learning_rate":learning_rate,"epochs":epochs,"mae":mae,"rmse":rmse,"r2":r2,"weights":w.tolist(),"imputer_median":medians.tolist(),"scaler_mean":mean.tolist(),"scaler_std":std.tolist(),"test_predictions":pred_rows}
    save_json(metrics, output_dir/'regression_metrics.json')
    plt.figure(figsize=(7,4.2)); plt.plot(loss); plt.xlabel('Epoch'); plt.ylabel('Training MSE'); plt.title('Regression Loss History'); plt.tight_layout(); plt.savefig(output_dir/'regression_loss.png', dpi=160); plt.close()
    np.savez(model_dir/'regression_model.npz', weights=w, imputer_median=medians, scaler_mean=mean, scaler_std=std, learning_rate=learning_rate, epochs=epochs)
    return metrics
