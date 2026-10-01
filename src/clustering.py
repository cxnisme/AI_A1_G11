import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .utils import save_json

def train_clustering(X, ids, feature_cols, output_dir, model_dir, random_state=42, k_min=2, k_max=5):
    imputer=SimpleImputer(strategy="median"); Xi=imputer.fit_transform(X); scaler=StandardScaler(); Xs=scaler.fit_transform(Xi)
    scores={}; models={}
    for k in range(k_min,k_max+1):
        if k>=len(Xs): continue
        m=KMeans(n_clusters=k,n_init=20,random_state=random_state); labels=m.fit_predict(Xs); score=silhouette_score(Xs,labels); scores[str(k)]=float(score); models[k]=m
    if not scores:
        raise ValueError('Not enough records to evaluate the requested k range')
    best_k=max(scores,key=scores.get); best_k=int(best_k); model=models[best_k]; labels=model.labels_
    metrics={"random_state":random_state,"feature_columns":feature_cols,"k_scores":scores,"selected_k":best_k,"k_range_evaluated":[k_min,k_max],"records_labelled":int(len(labels)),"selection_rule":"highest silhouette score","interpretation_note":"Clusters are unsupervised groupings of the six input features only. They are not verified real-world categories and should be used as context, not as a decision rule."}
    save_json(metrics,output_dir/'clustering_metrics.json')
    pd.DataFrame({"record_id":ids,"cluster":labels}).to_csv(output_dir/'clusters.csv',index=False)
    # PCA-free plot: first two standardized input features, clearly labeled as a 2D projection of selected inputs
    plt.figure(figsize=(6.2,4.6)); plt.scatter(Xs[:,0],Xs[:,1],c=labels,cmap='viridis',s=55); plt.xlabel(f'{feature_cols[0]} (standardized)'); plt.ylabel(f'{feature_cols[1]} (standardized)'); plt.title(f'Cluster View (k={best_k})'); plt.tight_layout(); plt.savefig(output_dir/'cluster_plot.png',dpi=160); plt.close()
    joblib.dump({"imputer":imputer,"scaler":scaler,"kmeans":model,"feature_columns":feature_cols,"selected_k":best_k},model_dir/'clustering_model.joblib')
    return metrics
