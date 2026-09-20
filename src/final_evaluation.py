"""One-time frozen held-out evaluation against legitimate non-oracle baselines."""
from __future__ import annotations

import json
import math
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import ARTIFACTS, RESULTS, SEEDS
from .data import load_bundle
from .metrics import classification_metrics

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TRAIN_CAP = 60_000
RBF_CAP = 15_000
GB_CAP = 30_000

MODEL_NAMES = [
    "Logistic Regression", "Linear SVM", "RBF SVM", "Gaussian Naive Bayes",
    "k-nearest neighbors", "Decision Tree", "Random Forest", "ExtraTrees",
    "Gradient Boosting", "HistGradientBoosting", "XGBoost", "LightGBM",
    "CatBoost", "MLP", "Graph-context MLP", "AnytimeGraph-Guard",
]


def make_model(name, seed, n_classes):
    common_weight = "balanced"
    if name == "Logistic Regression": return LogisticRegression(max_iter=500, C=1, class_weight=common_weight, random_state=seed)
    if name == "Linear SVM": return LinearSVC(C=1, class_weight=common_weight, random_state=seed)
    if name == "RBF SVM": return SVC(C=5, gamma="scale", class_weight=common_weight, random_state=seed)
    if name == "Gaussian Naive Bayes": return GaussianNB(var_smoothing=1e-8)
    if name == "k-nearest neighbors": return KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)
    if name == "Decision Tree": return DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, class_weight=common_weight, random_state=seed)
    if name == "Random Forest": return RandomForestClassifier(n_estimators=180, max_features="sqrt", min_samples_leaf=1, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "ExtraTrees": return ExtraTreesClassifier(n_estimators=220, max_features="sqrt", min_samples_leaf=1, class_weight=common_weight, n_jobs=-1, random_state=seed)
    if name == "Gradient Boosting": return GradientBoostingClassifier(n_estimators=120, learning_rate=.06, max_depth=3, random_state=seed)
    if name == "HistGradientBoosting": return HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=63, learning_rate=.06, l2_regularization=2, min_samples_leaf=30, class_weight=common_weight, random_state=seed)
    if name == "XGBoost": return XGBClassifier(n_estimators=260, max_depth=8, learning_rate=.06, subsample=.9, colsample_bytree=.9, objective="multi:softprob", eval_metric="mlogloss", n_jobs=-1, random_state=seed)
    if name == "LightGBM": return LGBMClassifier(n_estimators=260, num_leaves=63, learning_rate=.05, subsample=.9, colsample_bytree=.9, class_weight=common_weight, n_jobs=-1, random_state=seed, verbosity=-1)
    if name == "CatBoost": return CatBoostClassifier(iterations=260, depth=8, learning_rate=.06, loss_function="MultiClass", auto_class_weights="Balanced", random_seed=seed, verbose=False, allow_writing_files=False, thread_count=-1)
    if name in {"MLP", "Graph-context MLP"}: return MLPClassifier(hidden_layer_sizes=(96,48), alpha=1e-4, batch_size=512, learning_rate_init=1e-3, early_stopping=True, validation_fraction=.15, max_iter=60, random_state=seed)
    if name == "AnytimeGraph-Guard": return HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=63, learning_rate=.06, l2_regularization=2, min_samples_leaf=30, class_weight=common_weight, random_state=seed)
    raise KeyError(name)


def stratified_cap(y, cap, seed):
    if len(y)<=cap: return np.arange(len(y))
    rng=np.random.default_rng(seed); idx=[]
    for c in np.unique(y):
        pool=np.flatnonzero(y==c); take=max(2,int(round(cap*len(pool)/len(y))))
        idx.extend(rng.choice(pool,min(take,len(pool)),replace=False).tolist())
    rng.shuffle(idx); return np.asarray(idx[:cap])


def probabilities(model,x,n_classes):
    if hasattr(model,"predict_proba"):
        p=model.predict_proba(x)
    else:
        d=model.decision_function(x)
        if d.ndim==1: d=np.column_stack([-d,d])
        p=softmax(d,axis=1)
    # Some capped learners can omit a class; align by model.classes_.
    if p.shape[1] != n_classes:
        out=np.full((len(x),n_classes),1e-12)
        for j,c in enumerate(model.classes_): out[:,int(c)]=p[:,j]
        p=out/out.sum(1,keepdims=True)
    p=np.clip(p,1e-9,1-1e-9)
    return p/p.sum(axis=1,keepdims=True)


def temperature_fit(y,p):
    logits=np.log(np.clip(p,1e-9,1))
    def nll(logt):
        q=softmax(logits/np.exp(logt),axis=1)
        return -np.log(q[np.arange(len(y)),y]+1e-12).mean()
    return float(np.exp(minimize_scalar(nll,bounds=(-2.3,2.3),method='bounded').x))


MACRO_MAPS={
 'toniot':{'normal':'benign','ddos':'flood','dos':'flood','scanning':'recon','backdoor':'access','password':'access','injection':'application','xss':'application','ransomware':'malware','mitm':'mitm'},
 'edgeiiot':{'Normal':'benign','DDoS_HTTP':'flood','DDoS_ICMP':'flood','DDoS_TCP':'flood','DDoS_UDP':'flood','Fingerprinting':'recon','Port_Scanning':'recon','Vulnerability_scanner':'recon','SQL_injection':'application','XSS':'application','Uploading':'application','Backdoor':'access','Password':'access','Ransomware':'malware','MITM':'mitm'},
 'apa_ddos':{'Benign':'benign','DDoS-ACK':'flood','DDoS-PSH-ACK':'flood'},
}


def aggregate_task(y,p,classes,mapping):
    names=sorted(set(mapping.values())); ni={n:i for i,n in enumerate(names)}
    cmap=np.asarray([ni[mapping[c]] for c in classes])
    yy=cmap[y]; pp=np.zeros((len(y),len(names)))
    for old,new in enumerate(cmap): pp[:,new]+=p[:,old]
    return yy,pp,names


def main():
    proto=ARTIFACTS/'protocol'/'frozen_protocol.json'; sel=ARTIFACTS/'protocol'/'frozen_selection.json'
    if not proto.exists() or not sel.exists(): raise RuntimeError('frozen protocol and selection required')
    selection=json.loads(sel.read_text(encoding='utf8'))
    if selection['selected_candidate']!='AnytimeGraph-Guard': raise RuntimeError('selection mismatch')
    eval_config={"models":MODEL_NAMES,"train_cap":TRAIN_CAP,"rbf_cap":RBF_CAP,"gradient_boost_cap":GB_CAP,"seeds":SEEDS,"feature_rule":"tabular baselines use X_base; graph models and proposed use X_all","created_before_test_load":True}
    (ARTIFACTS/'protocol'/'final_evaluation_config.json').write_text(json.dumps(eval_config,indent=2),encoding='utf8')
    records=[]; perclass=[]; confusions=[]; runtime=[]
    for dataset in ['toniot','edgeiiot','apa_ddos']:
      b=load_bundle(dataset); nc=len(b.classes); benign=b.classes.index({'toniot':'normal','edgeiiot':'Normal','apa_ddos':'Benign'}[dataset])
      for seed in SEEDS:
        common_idx=stratified_cap(b.y['train'],TRAIN_CAP,seed)
        for name in MODEL_NAMES:
          graph=name in {'Graph-context MLP','AnytimeGraph-Guard'}
          X=b.X_all if graph else b.X_base
          cap=RBF_CAP if name=='RBF SVM' else GB_CAP if name=='Gradient Boosting' else TRAIN_CAP
          idx=stratified_cap(b.y['train'],cap,seed)
          model=make_model(name,seed,nc)
          start=time.perf_counter(); model.fit(X['train'][idx],b.y['train'][idx]); train_s=time.perf_counter()-start
          start=time.perf_counter(); pv=probabilities(model,X['validation'],nc); pred_s=time.perf_counter()-start
          start=time.perf_counter(); pt=probabilities(model,X['test'],nc); infer_s=time.perf_counter()-start
          temp=temperature_fit(b.y['validation'],pv) if name=='AnytimeGraph-Guard' else 1.0
          if temp!=1: pt=softmax(np.log(pt)/temp,axis=1)
          for task in ['multiclass','binary','macro']:
            if task=='multiclass': yy,pp,names=b.y['test'],pt,b.classes
            elif task=='binary':
              yy=b.y_binary['test']; attack=1-pt[:,benign]; pp=np.column_stack([1-attack,attack]); names=['benign','attack']
            else: yy,pp,names=aggregate_task(b.y['test'],pt,b.classes,MACRO_MAPS[dataset])
            m=classification_metrics(yy,pp)
            records.append({'dataset':dataset,'model':name,'seed':seed,'task':task,**m,'temperature':temp})
            if task=='multiclass':
              pred=pp.argmax(1); rec=recall_score(yy,pred,labels=np.arange(len(names)),average=None,zero_division=0)
              for c,val in zip(names,rec): perclass.append({'dataset':dataset,'model':name,'seed':seed,'class':c,'recall':val})
              cm=confusion_matrix(yy,pred,labels=np.arange(len(names)),normalize='true')
              for i,a in enumerate(names):
                for j,z in enumerate(names): confusions.append({'dataset':dataset,'model':name,'seed':seed,'true':a,'predicted':z,'value':cm[i,j]})
          runtime.append({'dataset':dataset,'model':name,'seed':seed,'train_seconds':train_s,'validation_predict_seconds':pred_s,'test_inference_seconds':infer_s,'test_rows':len(b.y['test']),'latency_ms_per_row':1000*infer_s/len(b.y['test'])})
    pd.DataFrame(records).to_csv(RESULTS/'final_test_metrics.csv',index=False)
    pd.DataFrame(perclass).to_csv(RESULTS/'per_class_recall.csv',index=False)
    pd.DataFrame(confusions).to_csv(RESULTS/'confusion_matrices.csv',index=False)
    pd.DataFrame(runtime).to_csv(RESULTS/'runtime.csv',index=False)


if __name__=='__main__': main()
