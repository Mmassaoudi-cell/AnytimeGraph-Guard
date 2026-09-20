"""Candidate-specific anytime evidence and reconstructed stopping evaluation."""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy.special import betaln
from scipy.stats import beta as beta_dist
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import PRIMARY_FILES, RESULTS, SEEDS
from .data import load_bundle
from .final_evaluation import probabilities, stratified_cap


ALPHA=.05
HORIZON=500
DATASETS=['toniot','edgeiiot','apa_ddos']
# Only APA-DDoS holds out disjoint source devices with a genuine sequential
# stream. ToN-IoT groups are directed endpoint pairs and Edge-IIoT groups are
# class-block capture segments (Table III); within each group, ascending row
# order is a proxy for temporal order, not a verified continuous capture, so
# their sequential results are reported as a weaker-ordering extension of the
# anytime evidence, not a second source-disjoint stream control.
STREAM_BASIS={
    'apa_ddos':'source-disjoint stream (verified ordering)',
    'toniot':'within-group row-order proxy (directed endpoint pair)',
    'edgeiiot':'within-group row-order proxy (capture segment)',
}
DRIFT_SEVERITIES=[0.0,0.5,1.0,2.0]  # additive shift in units of one calibration-benign SD
DRIFT_ABS_FLOOR=0.01  # absolute attack-probability floor used when calibration SD is ~0 (e.g. APA)


def online_e(scores, calibration, seed, kappa=.5):
    """Randomized online-rank conformal betting process.

    Under exchangeable benign scores, sequential insertion ranks are uniform;
    f(p)=(1-kappa)p^-kappa integrates to one. The cumulative product is an
    e-process. We work in log space and preserve the exchangeability caveat.
    """
    rng=np.random.default_rng(seed); history=list(np.asarray(calibration,float)); loge=[]; total=0.0; pvals=[]
    for s in np.asarray(scores,float):
        h=np.asarray(history); greater=(h>s).sum(); equal=(h==s).sum()
        p=(greater+rng.random()*(equal+1))/(len(h)+1); p=max(p,1e-12)
        total += math.log(1-kappa)-kappa*math.log(p)
        loge.append(total); pvals.append(p); history.append(float(s))
    return np.asarray(loge),np.asarray(pvals)


def first_crossing(stat,threshold):
    hit=np.flatnonzero(np.asarray(stat)>=threshold); return int(hit[0]+1) if len(hit) else None


def beta_params(x):
    x=np.clip(np.asarray(x),1e-5,1-1e-5); m=x.mean(); v=max(x.var(),1e-5)
    q=max(2,m*(1-m)/v-1); return max(.2,m*q),max(.2,(1-m)*q)


def beta_logpdf(x,a,b):
    x=np.clip(x,1e-6,1-1e-6); return (a-1)*np.log(x)+(b-1)*np.log(1-x)-betaln(a,b)


def q_telemetry_policy(base,graph,y):
    """Validation-trained cost-sensitive telemetry heuristic over observe/inspect/expand.

    This is a single-step, state-conditioned action-value table -- a bandit-
    style heuristic, not a multi-step reinforcement-learning agent. State:
    five risk bins x five disagreement bins. The reward is correct attack
    belief minus telemetry cost; this is explicitly reconstructed because no
    logged analyst or containment outcome exists in the source corpora.
    """
    q=np.zeros((5,5,3)); counts=np.zeros_like(q)
    costs=np.array([0.0,.02,.05])
    for pb,pg,yy in zip(base,graph,y):
        s=(min(4,int(pb*5)),min(4,int(abs(pb-pg)*10)))
        obs=np.array([pb,(pb+pg)/2,pg]); reward=1-np.abs(obs-yy)-costs
        q[s]+=reward; counts[s]+=1
    q=np.divide(q,counts,out=np.zeros_like(q),where=counts>0)
    return q


def policy_scores(base,graph,q):
    out=[]; actions=[]
    for pb,pg in zip(base,graph):
        s=(min(4,int(pb*5)),min(4,int(abs(pb-pg)*10))); a=int(q[s].argmax())
        out.append([pb,(pb+pg)/2,pg][a]); actions.append(['observe','inspect_stream','expand_graph'][a])
    return np.asarray(out),actions


def detection_stats(method,scores,cal0,cal1,seed,cusum_h=None,window_thr=None):
    if method=='AnytimeGraph-Guard':
        loge,p=online_e(scores,cal0,seed); d=first_crossing(loge,math.log(1/ALPHA)); return d,loge
    if method=='fixed_threshold':
        d=first_crossing(scores,np.quantile(cal0,1-ALPHA)); return d,scores
    if method=='fixed_window':
        w=20; stat=pd.Series(scores).rolling(w,min_periods=w).mean().fillna(0).to_numpy(); d=first_crossing(stat,window_thr); return d,stat
    if method=='CUSUM':
        mu=np.mean(cal0); k=.5*np.std(cal0); vals=[]; z=0
        for s in scores: z=max(0,z+s-mu-k); vals.append(z)
        d=first_crossing(vals,cusum_h); return d,np.asarray(vals)
    if method=='SPRT':
        a0,b0=beta_params(cal0); a1,b1=beta_params(cal1); lr=np.cumsum(beta_logpdf(scores,a1,b1)-beta_logpdf(scores,a0,b0)); d=first_crossing(lr,math.log((1-.2)/ALPHA)); return d,lr
    if method=='single_conformal':
        cal=np.sort(cal0); p=(len(cal)-np.searchsorted(cal,scores,side='left')+1)/(len(cal)+1); d=first_crossing(-p,-ALPHA); return d,-p
    raise KeyError(method)


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05):
    """Exact binomial confidence interval for an empirical Type-I/detection rate."""
    if trials == 0:
        return float('nan'), float('nan')
    lo = 0.0 if successes == 0 else float(beta_dist.ppf(alpha/2, successes, trials-successes+1))
    hi = 1.0 if successes == trials else float(beta_dist.ppf(1-alpha/2, successes+1, trials-successes))
    return lo, hi


def main():
    methods=['AnytimeGraph-Guard','fixed_threshold','fixed_window','CUSUM','SPRT','single_conformal']
    rows=[]; trajectories=[]; actions_rows=[]; drift_rows=[]
    for dataset in DATASETS:
      b=load_bundle(dataset)
      for seed in SEEDS:
        idx=stratified_cap(b.y['train'],60000,seed)
        full=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=63,learning_rate=.06,l2_regularization=2,min_samples_leaf=30,class_weight='balanced',random_state=seed).fit(b.X_all['train'][idx],b.y_binary['train'][idx])
        base=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=63,learning_rate=.06,l2_regularization=2,min_samples_leaf=30,class_weight='balanced',random_state=seed).fit(b.X_base['train'][idx],b.y_binary['train'][idx])
        gv=probabilities(full,b.X_all['validation'],2)[:,1]; bv=probabilities(base,b.X_base['validation'],2)[:,1]
        gt=probabilities(full,b.X_all['test'],2)[:,1]; bt=probabilities(base,b.X_base['test'],2)[:,1]
        cal0=gv[b.y_binary['validation']==0]; cal1=gv[b.y_binary['validation']==1]
        if len(cal0)<5 or len(cal1)<5:
          continue
        q=q_telemetry_policy(bv,gv,b.y_binary['validation']); policy,acts=policy_scores(bt,gt,q)
        # Validation-only nuisance thresholds.
        roll=pd.Series(cal0).rolling(20,min_periods=20).mean().dropna()
        window_thr=float(roll.quantile(.95)) if len(roll) else float(np.quantile(cal0,.95))
        chunks=np.array_split(cal0,max(1,len(cal0)//HORIZON)); maxima=[]
        for ch in chunks:
          z=0; vals=[]; mu=cal0.mean(); k=.5*cal0.std()
          for s in ch: z=max(0,z+s-mu-k); vals.append(z)
          maxima.append(max(vals) if vals else 0)
        cusum_h=float(np.quantile(maxima,.95))
        cal_sd=float(cal0.std())
        test_meta=pd.DataFrame({'group':b.groups['test'],'y':b.y_binary['test'],'score':policy,'raw_graph':gt,'action':acts,'row':b.row_ids['test']})
        for group,g in test_meta.groupby('group'):
          g=g.sort_values('row'); yy=int(round(g.y.mean())); score=g.score.to_numpy()
          windows=[score[i:i+HORIZON] for i in range(0,len(score),HORIZON) if len(score[i:i+HORIZON])>=100] if yy==0 else [score[:HORIZON]]
          for widx,episode in enumerate(windows):
            for method in methods:
              delay,stat=detection_stats(method,episode,cal0,cal1,seed+widx,cusum_h,window_thr)
              rows.append({'dataset':dataset,'seed':seed,'group':group,'episode':widx,'true_attack':yy,'method':method,'alarm':int(delay is not None),'delay_flows':delay if delay is not None else len(episode)+1,'average_sample_number':delay if delay is not None else len(episode),'episode_length':len(episode)})
            if widx==0:
              loge,p=online_e(episode,cal0,seed)
              for t,(sc,le) in enumerate(zip(episode,loge),1): trajectories.append({'dataset':dataset,'seed':seed,'group':group,'true_attack':yy,'t':t,'score':sc,'log_e':le,'threshold':math.log(1/ALPHA)})
              # Drift-injection stress test: additive shifts to a benign
              # episode's scores simulate benign-score drift away from the
              # calibration distribution; severity 0.0 is the unperturbed
              # exchangeable case. APA's calibration benign scores are
              # essentially degenerate (std approx 0, near-perfect
              # separability), so a fixed absolute floor stands in for the
              # unit step there instead of a meaningless zero-width SD.
              if yy==0:
                step=cal_sd if cal_sd>1e-6 else DRIFT_ABS_FLOOR
                for severity in DRIFT_SEVERITIES:
                  shifted=np.clip(episode+severity*step,0,1)
                  loge_d,_=online_e(shifted,cal0,seed)
                  alarmed=int(bool(np.any(loge_d>=math.log(1/ALPHA))))
                  drift_rows.append({'dataset':dataset,'seed':seed,'group':group,'shift_calibration_sd':severity,
                                      'shift_unit':'calibration_sd' if cal_sd>1e-6 else 'absolute_floor',
                                      'alarmed':alarmed})
        for action,count in pd.Series(acts).value_counts().items(): actions_rows.append({'dataset':dataset,'seed':seed,'action':action,'count':int(count)})

    detail=pd.DataFrame(rows); detail.to_csv(RESULTS/'sequential_episode_results.csv',index=False)
    summary=[]
    for (dataset,method),g in detail.groupby(['dataset','method']):
      benign=g[g.true_attack==0]; attack=g[g.true_attack==1]
      n_benign=len(benign); n_attack=len(attack); alarms=int(benign.alarm.sum())
      type_i_lo,type_i_hi=clopper_pearson(alarms,n_benign)
      det_alarms=int(attack.alarm.sum())
      det_lo,det_hi=clopper_pearson(det_alarms,n_attack)
      summary.append({'dataset':dataset,'method':method,'benign_windows':n_benign,
                       'empirical_type_i':benign.alarm.mean() if n_benign else float('nan'),
                       'type_i_ci95_low':type_i_lo,'type_i_ci95_high':type_i_hi,
                       'attack_episodes':n_attack,
                       'detection_rate':attack.alarm.mean() if n_attack else float('nan'),
                       'detection_rate_ci95_low':det_lo,'detection_rate_ci95_high':det_hi,
                       'mean_detection_delay':attack.loc[attack.alarm==1,'delay_flows'].mean(),
                       'average_sample_number':g.average_sample_number.mean(),'alpha':ALPHA,
                       'stream_basis':STREAM_BASIS[dataset]})
    pd.DataFrame(summary).to_csv(RESULTS/'sequential_summary.csv',index=False)
    pd.DataFrame(trajectories).to_csv(RESULTS/'eprocess_trajectories.csv',index=False)
    pd.DataFrame(actions_rows).to_csv(RESULTS/'policy_action_counts.csv',index=False)

    drift=pd.DataFrame(drift_rows)
    drift_summary=drift.groupby(['dataset','shift_calibration_sd']).agg(
        benign_windows=('alarmed','size'), alarms=('alarmed','sum')).reset_index()
    drift_summary['empirical_type_i']=drift_summary['alarms']/drift_summary['benign_windows']
    ci=drift_summary.apply(lambda r: clopper_pearson(int(r['alarms']), int(r['benign_windows'])), axis=1)
    drift_summary['ci95_low']=[c[0] for c in ci]; drift_summary['ci95_high']=[c[1] for c in ci]
    drift_summary.to_csv(RESULTS/'eprocess_drift_stress.csv',index=False)


if __name__=='__main__':main()
