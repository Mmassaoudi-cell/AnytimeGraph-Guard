"""Stage-A smoke tests and Stage-B validation-only screening."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .candidates import HDGraphClassifier, ParameterMatchedMLP, RelationKAN, torch_proba, train_torch
from .config import ARTIFACTS, REPORTS, RESULTS, SEEDS
from .data import load_bundle
from .metrics import classification_metrics


FEASIBILITY = [
    (1, "RelKAN-Guard", "fully supported", "All three selected corpora expose relations and endpoint graphs; relation-conditioned spline messages can be evaluated directly."),
    (2, "KFE-Guard", "unsupported", "No corpus provides validated future graph-state densities, critical-host compromise horizons, or action-conditioned transitions."),
    (3, "AnytimeGraph-Guard", "fully supported", "APA and raw Edge events are ordered; APA provides source-disjoint benign/attack streams for anytime evidence and stopping."),
    (4, "ParticleGuard-GRL", "partially supported", "Missing/deceptive telemetry and hidden attacker states must be simulated; particle belief accuracy can be tested only under declared masks."),
    (5, "WasserGuard-GRL", "partially supported", "Structured feature/protocol/intensity perturbations are feasible, but observed attacker action distributions and rewards are absent."),
    (6, "CondenseGuard-RL", "partially supported", "Graphs can be condensed, but logged response outcomes and Bellman targets are absent; only detector/rare-class retention is testable."),
    (7, "AutoEdge-Guard", "partially supported", "Desktop CPU/GPU timings support a hardware-proxy search only; no target edge device or measured energy is available."),
    (8, "HD-Guard-RL", "fully supported", "Endpoint, relation, and feature hypervectors plus online prototype updates are directly implementable; action quality remains simulated."),
    (9, "SINDyGraph-Guard", "unsupported", "No defensible continuous attack-state derivatives, containment inputs, or controlled propagation equations are present."),
    (10, "InvariantGuard-RL", "partially supported", "Multiple corpora form environments, but only binary labels and coarse graph statistics reconcile without inventing mechanisms."),
]


def subset(x, y, r, cap, seed):
    if len(y) <= cap: return x, y, r
    rng=np.random.default_rng(seed); idx=[]
    for c in np.unique(y):
        pool=np.flatnonzero(y==c); take=max(10,int(cap*len(pool)/len(y)))
        idx.extend(rng.choice(pool,size=min(take,len(pool)),replace=False))
    idx=np.asarray(idx[:cap]); return x[idx],y[idx],r[idx]


def main():
    rows=[]; smoke=[]
    for name in ["toniot","edgeiiot","apa_ddos"]:
        bundle=load_bundle(name)
        n_classes=len(bundle.classes)
        for seed in SEEDS[:3]:
            xtr,ytr,rtr=subset(bundle.X_all["train"],bundle.y["train"],bundle.relation["train"],30000,seed)
            xva,yva,rva=subset(bundle.X_all["validation"],bundle.y["validation"],bundle.relation["validation"],20000,seed+1)
            # Candidate 3 graph evidence encoder (test data is untouched).
            start=time.perf_counter()
            hgb=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=31,learning_rate=.08,l2_regularization=1,random_state=seed).fit(xtr,ytr)
            p=hgb.predict_proba(xva); elapsed=time.perf_counter()-start
            m=classification_metrics(yva,p); rows.append({"candidate":"AnytimeGraph-Guard","dataset":name,"seed":seed,**m,"train_seconds":elapsed,"parameters":-1})
            # Candidate 1 relation-conditioned KAN.
            kan=RelationKAN(xtr.shape[1],int(max(rtr.max(),rva.max())),n_classes,grid=8)
            kan,elapsed=train_torch(kan,xtr,ytr,rtr,seed,epochs=8,lr=2e-3,reg=2e-3)
            p=torch_proba(kan,xva,rva); km=classification_metrics(yva,p)
            rows.append({"candidate":"RelKAN-Guard","dataset":name,"seed":seed,**km,"train_seconds":elapsed,"parameters":sum(v.numel() for v in kan.parameters())})
            # Candidate 8 graph hypervectors.
            start=time.perf_counter(); hd=HDGraphClassifier(dim=1024,seed=seed).fit(xtr,ytr); elapsed=time.perf_counter()-start
            p=hd.predict_proba(xva); m=classification_metrics(yva,p)
            rows.append({"candidate":"HD-Guard-RL","dataset":name,"seed":seed,**m,"train_seconds":elapsed,"parameters":xtr.shape[1]*1024+n_classes*1024})
            # Parameter-matched mechanism check for RelKAN.
            mlp=ParameterMatchedMLP(xtr.shape[1],n_classes,hidden=max(16,min(64,(sum(v.numel() for v in kan.parameters())//max(1,xtr.shape[1]+n_classes)))))
            mlp,te=train_torch(mlp,xtr,ytr,rtr,seed,epochs=8,lr=2e-3)
            pm=torch_proba(mlp,xva,rva); mm=classification_metrics(yva,pm)
            smoke.append({"dataset":name,"seed":seed,"mechanism":"RelKAN vs parameter-matched MLP","relkan_macro_f1":km["macro_f1"],"mlp_macro_f1":mm["macro_f1"]})
    df=pd.DataFrame(rows); df.to_csv(RESULTS/"candidate_validation_runs.csv",index=False)
    pd.DataFrame(smoke).to_csv(RESULTS/"candidate_mechanism_smoke.csv",index=False)
    summary=df.groupby(["candidate","dataset"]).agg({"macro_f1":["median","min","std"],"minority_recall":"median","macro_pr_auc":"median","ece":"median","train_seconds":"median"}).reset_index()
    summary.columns=['candidate','dataset','macro_f1_median','macro_f1_worst','macro_f1_std','minority_recall_median','pr_auc_median','ece_median','train_seconds_median']
    summary.to_csv(RESULTS/"candidate_validation_summary.csv",index=False)
    rank=df.groupby('candidate').agg(macro_f1=('macro_f1','median'),worst=('macro_f1','min'),minority=('minority_recall','median'),pr_auc=('macro_pr_auc','median'),ece=('ece','median'),seconds=('train_seconds','median')).reset_index()
    # Constrained lexicographic score: collapse and minority failures dominate.
    rank['eligible_no_collapse']=rank.worst>=0.10
    rank['rank']=rank.sort_values(['eligible_no_collapse','macro_f1','minority','pr_auc'],ascending=[False,False,False,False]).reset_index().index+1
    rank=rank.sort_values(['eligible_no_collapse','macro_f1','minority','pr_auc'],ascending=[False,False,False,False])
    rank.to_csv(RESULTS/"candidate_validation_ranking.csv",index=False)
    payload={"feasibility":[{"id":i,"candidate":c,"status":s,"reason":r} for i,c,s,r in FEASIBILITY],"screened_candidates":["RelKAN-Guard","AnytimeGraph-Guard","HD-Guard-RL"],"partial_candidate_prototype_modules":["ParticleBelief","condensed_coreset","structured-shift generator in robustness evaluation","hardware-proxy compact search","binary multi-environment evaluation"],"selection_data":"validation only"}
    (ARTIFACTS/"candidate_feasibility.json").write_text(json.dumps(payload,indent=2),encoding='utf8')
    lines=['# Candidate Feasibility Assessment','', '| ID | Candidate | Status | Evidence-based reason |','|---:|---|---|---|']
    for i,c,s,r in FEASIBILITY: lines.append(f'| {i} | {c} | {s} | {r} |')
    lines += ['', 'Only candidates classified as fully supported enter the common three-seed Stage-B ranking. Partially supported mechanisms are retained for scoped smoke tests and negative findings; they cannot become the final method on simulated assumptions alone.']
    (REPORTS/"CANDIDATE_FEASIBILITY.md").write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__': main()
