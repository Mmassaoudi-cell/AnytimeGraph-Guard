"""Paired tests, bootstrap intervals, Holm correction, and W/T/L accounting."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from .config import RESULTS

DELTA=.02


def holm(p):
    p=np.asarray(p,float); order=np.argsort(p); out=np.empty_like(p); running=0
    n=len(p)
    for rank,idx in enumerate(order):
        val=min(1,(n-rank)*p[idx]); running=max(running,val); out[idx]=running
    return out


def minimum_detectable_effect(n, alpha=.05, power=.80):
    """Minimum detectable paired effect (Cohen's dz) for a two-sided t-test.

    Solved exactly via the noncentral-t power function rather than the usual
    normal-approximation shortcut, since the headline comparisons use n=5.
    """
    df=n-1
    t_crit=stats.t.ppf(1-alpha/2,df)
    def power_at_ncp(ncp):
        return 1-stats.nct.cdf(t_crit,df,ncp)+stats.nct.cdf(-t_crit,df,ncp)
    upper=9.0  # scipy's noncentral-t cdf becomes numerically unstable above this
    if power_at_ncp(upper)<power:
        return float('nan')
    ncp=brentq(lambda z: power_at_ncp(z)-power,1e-6,upper)
    return float(ncp/math.sqrt(n))


def paired_record(group,baseline,diff):
    d=np.asarray(diff,float); n=len(d); mean=float(d.mean()); sd=float(d.std(ddof=1)) if n>1 else 0
    if sd==0:
        p_t=0.0 if mean!=0 else 1.0; p_equiv=0.0 if abs(mean)<DELTA else 1.0; dz=math.copysign(float('inf'),mean) if mean else 0.0
    else:
        p_t=float(stats.ttest_1samp(d,0).pvalue)
        se=sd/math.sqrt(n)
        p_lower=float(1-stats.t.cdf((mean+DELTA)/se,n-1))
        p_upper=float(stats.t.cdf((mean-DELTA)/se,n-1))
        p_equiv=max(p_lower,p_upper); dz=mean/sd
    try: p_w=float(stats.wilcoxon(d,zero_method='zsplit').pvalue)
    except ValueError: p_w=1.0
    rng=np.random.default_rng(20260731); boot=np.array([rng.choice(d,n,replace=True).mean() for _ in range(10000)])
    return {**group,'baseline':baseline,'n_pairs':n,'mean_difference':mean,'sd_difference':sd,'ci95_low':float(np.quantile(boot,.025)),'ci95_high':float(np.quantile(boot,.975)),'paired_t_p':p_t,'wilcoxon_p':p_w,'tost_p':p_equiv,'cohen_dz':dz}


def main():
    f=pd.read_csv(RESULTS/'final_test_metrics.csv'); f=f[f.task=='multiclass']
    pivot=f.pivot_table(index=['dataset','seed'],columns='model',values='macro_f1')
    proposed=pivot['AnytimeGraph-Guard']; rec=[]
    for dataset in pivot.index.get_level_values(0).unique():
        p=pivot.loc[dataset]
        for baseline in p.columns:
            if baseline!='AnytimeGraph-Guard': rec.append(paired_record({'scope':'dataset','dataset':dataset},baseline,p['AnytimeGraph-Guard']-p[baseline]))
    for baseline in pivot.columns:
        if baseline!='AnytimeGraph-Guard': rec.append(paired_record({'scope':'aggregate','dataset':'all_primary'},baseline,proposed-pivot[baseline]))
    out=pd.DataFrame(rec)
    for scope,idx in out.groupby('scope').groups.items():
        out.loc[idx,'paired_t_p_holm']=holm(out.loc[idx,'paired_t_p'])
        out.loc[idx,'tost_p_holm']=holm(out.loc[idx,'tost_p'])
    def outcome(r):
        if r.tost_p_holm<.05 and abs(r.mean_difference)<DELTA: return 'statistical tie'
        if r.paired_t_p_holm<.05 and r.mean_difference>0: return 'statistically superior'
        if r.paired_t_p_holm<.05 and r.mean_difference<0: return 'inferior'
        if r.mean_difference>DELTA: return 'mean-superior'
        if r.mean_difference<-DELTA: return 'inferior'
        return 'practical tie'
    out['outcome']=out.apply(outcome,axis=1)
    out.to_csv(RESULTS/'statistical_tests.csv',index=False)
    agg=out[out.scope=='aggregate'].copy()
    counts=agg.outcome.value_counts().to_dict()
    # A benchmark is "beaten" only for positive mean difference; statistical
    # and mean-superior outcomes are reported separately.
    summary={
      'statistically_superior':int((agg.outcome=='statistically superior').sum()),
      'mean_superior':int((agg.outcome=='mean-superior').sum()),
      'statistical_ties':int((agg.outcome=='statistical tie').sum()),
      'practical_ties':int((agg.outcome=='practical tie').sum()),
      'losses':int((agg.outcome=='inferior').sum()),
      'benchmarks_with_higher_aggregate_mean':int((agg.mean_difference<0).sum()),
      'benchmarks_beaten_by_aggregate_mean':int((agg.mean_difference>0).sum()),
      'total_real_benchmarks':len(agg),
    }
    pd.DataFrame([summary]).to_csv(RESULTS/'win_tie_loss_summary.csv',index=False)

    # Minimum detectable effect at alpha=.05, power=.80, expressed in Cohen's
    # dz and translated to an approximate macro-F1 scale using the SD
    # actually observed in the corresponding paired comparison.
    power_rows=[]
    headline_sd=float(out.loc[out.scope=='aggregate','sd_difference'].median())
    power_rows.append({'n_pairs':5,'comparison':'headline five-seed comparisons (per baseline)',
                        'alpha':.05,'power':.80,'mde_cohens_dz':minimum_detectable_effect(5),
                        'reference_sd_macro_f1':headline_sd,
                        'approx_mde_macro_f1':minimum_detectable_effect(5)*headline_sd})
    enh_path=RESULTS/'enhancement_results.csv'
    if enh_path.exists():
        en=pd.read_csv(enh_path)
        piv=en.pivot(index=['dataset','seed'],columns='model',values='macro_f1')
        enh_delta=piv['Validation-gated dual booster']-piv['Frozen base']
        enh_sd=float(enh_delta.std(ddof=1)); n_enh=int(len(enh_delta))
        dz15=minimum_detectable_effect(n_enh)
        power_rows.append({'n_pairs':n_enh,'comparison':'pooled enhancement comparison (3 datasets x 5 seeds)',
                            'alpha':.05,'power':.80,'mde_cohens_dz':dz15,
                            'reference_sd_macro_f1':enh_sd,'approx_mde_macro_f1':dz15*enh_sd})
    pd.DataFrame(power_rows).to_csv(RESULTS/'power_analysis.csv',index=False)


if __name__=='__main__':main()
