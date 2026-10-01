from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from .utils import save_json

def train_classifier(X, y, feature_cols, output_dir, model_dir, random_state=42, threshold=0.5):
    if len(set(y.tolist()))<2:
        raise ValueError('dispatch_attention must contain both classes 0 and 1 in the labelled rows to train the classifier')
    stratify=y if len(set(y))==2 and min((y==0).sum(), (y==1).sum())>=2 else None
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=random_state,stratify=stratify)
    pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',LogisticRegression(max_iter=1000,random_state=random_state))])
    pipe.fit(X_train,y_train); prob=pipe.predict_proba(X_test)[:,1]; pred=(prob>=threshold).astype(int)
    cm=confusion_matrix(y_test,pred,labels=[0,1])
    metrics={"model_type":"LogisticRegression","random_state":random_state,"test_size":0.2,"decision_threshold":float(threshold),"train_rows":int(len(y_train)),"test_rows":int(len(y_test)),"scaler_fit_on":"training split only","feature_columns":feature_cols,"stratified":stratify is not None,"accuracy":float(accuracy_score(y_test,pred)),"precision":float(precision_score(y_test,pred,zero_division=0)),"recall":float(recall_score(y_test,pred,zero_division=0)),"f1":float(f1_score(y_test,pred,zero_division=0)),"confusion_matrix":cm.tolist(),"error_cost_note":"A false negative (predicting 0 when the consignment needs attention) is usually more costly here: a problem consignment is dispatched without review. A false positive only costs staff review time. Lowering the decision threshold trades more false positives for fewer false negatives."}
    save_json(metrics, output_dir/'classification_metrics.json')
    plt.figure(figsize=(5.2,4.2)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False,xticklabels=['0','1'],yticklabels=['0','1']); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Dispatch Attention Confusion Matrix'); plt.tight_layout(); plt.savefig(output_dir/'confusion_matrix.png',dpi=160); plt.close()
    joblib.dump(pipe, model_dir/'classification_model.joblib')
    save_json({'decision_threshold':float(threshold)}, model_dir/'classification_config.json')
    return metrics
