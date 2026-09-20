"""Ablation, low-label, structured robustness, and calibration experiments."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import RESULTS, SEEDS
from .data import load_bundle
from .final_evaluation import make_model, probabilities, stratified_cap, temperature_fit
from .metrics import classification_metrics


def main():
    ablations=[]; low=[]; robust=[]; preds=[]; calibration=[]
    variants={
      'full':lambda b,s:b.X_all[s],
      'no_graph_context':lambda b,s:b.X_base[s],
      'graph_context_only':lambda b,s:b.X_graph[s],
      'no_relation_frequency':lambda b,s:np.delete(b.X_all[s],b.X_all[s].shape[1]-3,axis=1),
      'no_topology_counts':lambda b,s:np.column_stack([b.X_base[s],b.X_graph[s][:,[3]]]),
    }
    for dataset in ['toniot','edgeiiot','apa_ddos']:
      b=load_bundle(dataset); nc=len(b.classes)
      benign=b.classes.index({'toniot':'normal','edgeiiot':'Normal','apa_ddos':'Benign'}[dataset])
      for seed in SEEDS[:3]:
        idx=stratified_cap(b.y['train'],60000,seed)
        # Ablations.
        for variant,getx in variants.items():
          model=make_model('AnytimeGraph-Guard',seed,nc).fit(getx(b,'train')[idx],b.y['train'][idx])
          pv=probabilities(model,getx(b,'validation'),nc); temp=temperature_fit(b.y['validation'],pv)
          pt=probabilities(model,getx(b,'test'),nc); pt=np.exp(np.log(pt)/temp); pt/=pt.sum(1,keepdims=True)
          ablations.append({'dataset':dataset,'seed':seed,'variant':variant,**classification_metrics(b.y['test'],pt)})
        # Low-label curve.
        for frac in [.01,.05,.10,.25,.50,1.0]:
          cap=max(nc*5,int(60000*frac)); ids=stratified_cap(b.y['train'],cap,seed+31)
          model=make_model('AnytimeGraph-Guard',seed,nc).fit(b.X_all['train'][ids],b.y['train'][ids])
          pt=probabilities(model,b.X_all['test'],nc)
          low.append({'dataset':dataset,'seed':seed,'labeled_fraction':frac,'labeled_rows':len(ids),**classification_metrics(b.y['test'],pt)})
        # One clean model drives every test-time stress condition.
        model=make_model('AnytimeGraph-Guard',seed,nc).fit(b.X_all['train'][idx],b.y['train'][idx])
        pv=probabilities(model,b.X_all['validation'],nc); temp=temperature_fit(b.y['validation'],pv)
        rng=np.random.default_rng(seed)
        x=b.X_all['test']; conditions={'clean':x.copy()}
        xn=x.copy(); xn[:,:-6]+=rng.normal(0,.10,size=xn[:,:-6].shape); conditions['feature_noise_0.10']=xn
        xm=x.copy(); mask=rng.random(xm[:,:-6].shape)<.20; xm[:,:-6][mask]=0; conditions['missing_features_0.20']=xm
        xt=x.copy(); xt[:,-6:]=0; conditions['topology_removed']=xt
        xe=x.copy(); take=rng.random(len(xe))<.20; xe[take,-6:]=xe[rng.permutation(len(xe))[:take.sum()],-6:]; conditions['edge_context_corruption_0.20']=xe
        xp=x.copy(); xp[:,-3]=0; conditions['unseen_protocol_relation']=xp
        xi=x.copy(); intensity=[i for i,n in enumerate(b.feature_names_base) if any(k in n.lower() for k in ['rate','byte','pkt','packet','count','duration'])]
        if intensity: xi[:,intensity]*=.5
        conditions['attack_intensity_half']=xi
        for cond,xc in conditions.items():
          p=probabilities(model,xc,nc); p=np.exp(np.log(p)/temp); p/=p.sum(1,keepdims=True)
          robust.append({'dataset':dataset,'seed':seed,'condition':cond,**classification_metrics(b.y['test'],p)})
          if cond=='clean':
            pred=p.argmax(1); conf=p.max(1); attack=1-p[:,benign]
            for j in range(len(pred)):
              preds.append({'dataset':dataset,'seed':seed,'row_id':int(b.row_ids['test'][j]),'y':int(b.y['test'][j]),'pred':int(pred[j]),'confidence':float(conf[j]),'correct':int(pred[j]==b.y['test'][j]),'attack_probability':float(attack[j])})
            edges=np.linspace(0,1,16)
            for lo,hi in zip(edges[:-1],edges[1:]):
              m=(conf>lo)&(conf<=hi)
              if m.any(): calibration.append({'dataset':dataset,'seed':seed,'bin_low':lo,'bin_high':hi,'count':int(m.sum()),'mean_confidence':float(conf[m].mean()),'accuracy':float((pred[m]==b.y['test'][m]).mean())})
    pd.DataFrame(ablations).to_csv(RESULTS/'ablation_results.csv',index=False)
    pd.DataFrame(low).to_csv(RESULTS/'low_label_results.csv',index=False)
    pd.DataFrame(robust).to_csv(RESULTS/'robustness_results.csv',index=False)
    pd.DataFrame(preds).to_parquet(RESULTS/'proposed_predictions.parquet',index=False)
    pd.DataFrame(calibration).to_csv(RESULTS/'reliability_bins.csv',index=False)


if __name__=='__main__':main()
