"""Stage-C reproducible tuning of the three validation-screened candidates."""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .candidate_screening import subset
from .candidates import HDGraphClassifier, RelationKAN, torch_proba, train_torch
from .config import ARTIFACTS, RESULTS, SEEDS
from .data import load_bundle
from .metrics import classification_metrics


def objective(records):
    df=pd.DataFrame(records)
    med=df.macro_f1.median(); worst=df.macro_f1.min(); minority=df.minority_recall.median()
    feasible=bool(worst>=.10 and minority>=.05 and df.ece.median()<=.30)
    return (feasible,med,minority,df.macro_pr_auc.median(),-df.ece.median())


def eval_config(candidate,config,seeds):
    rows=[]
    for name in ['toniot','edgeiiot','apa_ddos']:
        b=load_bundle(name); n=len(b.classes)
        for seed in seeds:
            xtr,ytr,rtr=subset(b.X_all['train'],b.y['train'],b.relation['train'],config.get('train_cap',50000),seed)
            xva,yva,rva=subset(b.X_all['validation'],b.y['validation'],b.relation['validation'],config.get('val_cap',25000),seed+17)
            start=time.perf_counter()
            if candidate=='AnytimeGraph-Guard':
                model=HistGradientBoostingClassifier(random_state=seed,**{k:v for k,v in config.items() if k not in {'train_cap','val_cap'}}).fit(xtr,ytr)
                p=model.predict_proba(xva); params=-1
            elif candidate=='RelKAN-Guard':
                model=RelationKAN(xtr.shape[1],int(max(rtr.max(),rva.max())),n,grid=config['grid'])
                model,_=train_torch(model,xtr,ytr,rtr,seed,epochs=config['epochs'],lr=config['lr'],reg=config['reg'])
                p=torch_proba(model,xva,rva); params=sum(v.numel() for v in model.parameters())
            else:
                model=HDGraphClassifier(dim=config['dim'],seed=seed).fit(xtr,ytr); p=model.predict_proba(xva); params=xtr.shape[1]*config['dim']+n*config['dim']
            elapsed=time.perf_counter()-start
            rows.append({'candidate':candidate,'dataset':name,'seed':seed,'config':json.dumps(config,sort_keys=True),**classification_metrics(yva,p),'seconds':elapsed,'parameters':params})
    return rows


def main():
    spaces={
      'AnytimeGraph-Guard':[
        {'max_iter':160,'max_leaf_nodes':31,'learning_rate':.08,'l2_regularization':1.0,'min_samples_leaf':20},
        {'max_iter':240,'max_leaf_nodes':31,'learning_rate':.05,'l2_regularization':2.0,'min_samples_leaf':20},
        {'max_iter':200,'max_leaf_nodes':63,'learning_rate':.06,'l2_regularization':2.0,'min_samples_leaf':30},
        {'max_iter':260,'max_leaf_nodes':63,'learning_rate':.04,'l2_regularization':4.0,'min_samples_leaf':40},
      ],
      'RelKAN-Guard':[
        {'grid':8,'epochs':16,'lr':.002,'reg':.002,'train_cap':40000,'val_cap':25000},
        {'grid':12,'epochs':18,'lr':.0015,'reg':.001,'train_cap':40000,'val_cap':25000},
        {'grid':6,'epochs':20,'lr':.003,'reg':.005,'train_cap':40000,'val_cap':25000},
      ],
      'HD-Guard-RL':[{'dim':1024,'train_cap':50000,'val_cap':25000},{'dim':2048,'train_cap':50000,'val_cap':25000},{'dim':4096,'train_cap':50000,'val_cap':25000}],
    }
    search=[]; best={}
    for cand,configs in spaces.items():
        scored=[]
        for trial,config in enumerate(configs):
            rows=eval_config(cand,config,SEEDS[:2]); search.extend([dict(r,trial=trial,stage='search') for r in rows])
            scored.append((objective(rows),config,trial))
        scored.sort(key=lambda x:x[0],reverse=True); best[cand]={'objective':scored[0][0],'config':scored[0][1],'trial':scored[0][2]}
    pd.DataFrame(search).to_csv(RESULTS/'hyperparameter_search_log.csv',index=False)
    final=[]
    for cand,rec in best.items():
        final.extend(eval_config(cand,rec['config'],SEEDS))
    df=pd.DataFrame(final); df.to_csv(RESULTS/'top3_tuned_validation_runs.csv',index=False)
    agg=df.groupby('candidate').agg(macro_f1_median=('macro_f1','median'),macro_f1_worst=('macro_f1','min'),macro_f1_std=('macro_f1','std'),minority_recall_median=('minority_recall','median'),pr_auc_median=('macro_pr_auc','median'),ece_median=('ece','median'),seconds_median=('seconds','median')).reset_index()
    agg['no_collapse']=agg.macro_f1_worst>=.10
    agg['minority_constraint']=agg.minority_recall_median>=.05
    agg.to_csv(RESULTS/'top3_tuned_validation_summary.csv',index=False)
    eligible=agg[agg.no_collapse & agg.minority_constraint].sort_values(['macro_f1_median','minority_recall_median','pr_auc_median'],ascending=False)
    if len(eligible)==0: chosen=agg.sort_values('macro_f1_median',ascending=False).iloc[0]['candidate']
    else: chosen=eligible.iloc[0]['candidate']
    frozen={
      'selected_candidate':chosen,
      'selection_basis':'five-seed validation only; median primary metric with no-collapse and minority-recall constraints',
      'selected_hyperparameters':best[chosen]['config'],
      'all_best_configs':{k:v['config'] for k,v in best.items()},
      'test_evaluated_at_selection':False,
      'primary_metric':'multiclass macro-F1',
      'constraints':{'worst_macro_f1_min':.10,'median_minority_recall_min':.05,'median_ece_max':.30},
    }
    (ARTIFACTS/'protocol'/'frozen_selection.json').write_text(json.dumps(frozen,indent=2),encoding='utf8')


if __name__=='__main__': main()
