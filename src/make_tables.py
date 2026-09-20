"""Generate the 18 required IEEE-ready LaTeX tables from saved artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACTS, RESULTS, TABLES
from .data import load_bundle


def esc(x):
    s=str(x); repl={'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(repl.get(c,c) for c in s)


def tex(name,caption,label,columns,rows,align=None,resize=True,foot=None):
    align=align or 'l'+'c'*(len(columns)-1)
    body=[r'\begin{table}[t]',r'\centering',r'\caption{'+caption+r'}',r'\label{'+label+r'}',r'\scriptsize']
    if resize: body.append(r'\resizebox{\columnwidth}{!}{%')
    body += [r'\begin{tabular}{'+align+r'}',r'\toprule',' & '.join(map(esc,columns))+r' \\',r'\midrule']
    for row in rows: body.append(' & '.join(esc(v) for v in row)+r' \\')
    body += [r'\bottomrule',r'\end{tabular}']
    if resize: body.append('}')
    if foot: body.append(r'\vspace{1pt}\parbox{\columnwidth}{\tiny '+foot+'}')
    body.append(r'\end{table}')
    (TABLES/name).write_text('\n'.join(body)+'\n',encoding='utf8')


def tex_raw(name,caption,label,columns,rows,align=None,resize=True,foot=None):
    """Write a table whose cells already contain deliberate LaTeX markup."""
    align=align or 'l'+'c'*(len(columns)-1)
    body=[r'\begin{table}[t]',r'\centering',r'\caption{'+caption+r'}',r'\label{'+label+r'}',r'\scriptsize']
    if resize: body.append(r'\resizebox{\columnwidth}{!}{%')
    body += [r'\begin{tabular}{'+align+r'}',r'\toprule',' & '.join(columns)+r' \\',r'\midrule']
    for row in rows: body.append(' & '.join(map(str,row))+r' \\')
    body += [r'\bottomrule',r'\end{tabular}']
    if resize: body.append('}')
    if foot: body.append(r'\vspace{1pt}\parbox{\columnwidth}{\tiny '+foot+'}')
    body.append(r'\end{table}')
    (TABLES/name).write_text('\n'.join(body)+'\n',encoding='utf8')


def pm(mean,std): return f'{mean:.3f} \\(\\pm {std:.3f}\\)'


def result_pivot(task,metric='macro_f1'):
    f=pd.read_csv(RESULTS/'final_test_metrics.csv'); f=f[f.task==task]
    g=f.groupby(['model','dataset'])[metric].agg(['mean','std']).reset_index();
    out={}
    for m,sub in g.groupby('model'): out[m]={r.dataset:pm(r['mean'],r['std']) for _,r in sub.iterrows()}
    return out


def main():
    inv=json.load(open(ARTIFACTS/'audit'/'dataset_inventory.json',encoding='utf8'))
    rows=[]
    for r in inv['datasets']:
        rows.append([r['dataset'],r['file_count_all_formats'],f"{r['total_size_bytes']/1024**3:.2f}",f"{r['rows']:,}",'/'.join(r['label_columns']) or '--','Y' if r['graph_feasible'] else 'N','Y' if r['temporal_feasible'] else 'N'])
    tex('table01_dataset_screening.tex','Complete local dataset screening.','tab:screen',['Corpus','Files','GiB','Rows','Labels','Graph','Time'],rows,foot='Rows and labels are exact; expensive quality diagnostics on nonprimary raw archives are sampled as declared in the audit report.')

    selected=[]; meta={'toniot':('ToN-IoT',42,20569),'edgeiiot':('Edge-IIoT',61,814),'apa_ddos':('APA-DDoS',22,0)}
    for ds in ['toniot','edgeiiot','apa_ddos']:
        b=load_bundle(ds); counts=np.bincount(np.concatenate([b.y[s] for s in ['train','validation','test']]),minlength=len(b.classes)); label,predictors,duplicates=meta[ds]
        selected.append([label,f'{int(counts.sum()+duplicates):,}',predictors,len(b.classes),f'{int(counts.min()):,}',f'{counts.max()/counts.min():.1f}:1',f'{duplicates:,}'])
    tex('table02_selected_statistics.tex','Selected dataset size and imbalance after duplicate control.','tab:data',['Dataset','Raw rows','Pred.','Classes','Min. class','Max/min','Duplicates'],selected)

    sp=pd.read_csv(RESULTS/'split_summary.csv'); rows=[[r.dataset,r.split,f'{int(r.rows):,}',int(r.groups),int(r.removed_exact_duplicates),r.split_basis] for _,r in sp.iterrows()]
    tex('table03_split_protocol.tex','Frozen leakage-safe split protocol.','tab:split',['Dataset','Split','Rows','Groups','Removed','Basis'],rows)

    feas=json.load(open(ARTIFACTS/'candidate_feasibility.json',encoding='utf8'))['feasibility']; rows=[[r['id'],r['candidate'],r['status'],r['reason']] for r in feas]
    tex('table04_candidate_feasibility.tex','Feasibility of the ten predeclared candidates.','tab:feas',['ID','Candidate','Status','Reason'],rows,align='cllp{5.0cm}',resize=True)

    r=pd.read_csv(RESULTS/'candidate_validation_ranking.csv'); rows=[[x.candidate,f'{x.macro_f1:.3f}',f'{x.worst:.3f}',f'{x.minority:.3f}',f'{x.pr_auc:.3f}',f'{x.ece:.3f}',str(x.eligible_no_collapse)] for _,x in r.iterrows()]
    tex('table05_candidate_validation.tex','Three-seed validation screening.','tab:screening',['Candidate','Med. F1','Worst','Min. rec.','PR-AUC','ECE','Stable'],rows)

    r=pd.read_csv(RESULTS/'top3_tuned_validation_summary.csv'); rows=[[x.candidate,f'{x.macro_f1_median:.3f}',f'{x.macro_f1_worst:.3f}',f'{x.macro_f1_std:.3f}',f'{x.minority_recall_median:.3f}',f'{x.ece_median:.3f}',str(bool(x.no_collapse and x.minority_constraint))] for _,x in r.iterrows()]
    tex('table06_top3_tuning.tex','Five-seed tuned validation outcome.','tab:tuning',['Candidate','Med. F1','Worst','SD','Min. rec.','ECE','Eligible'],rows)

    sel=json.load(open(ARTIFACTS/'protocol'/'frozen_selection.json',encoding='utf8')); hp=sel['selected_hyperparameters']; rows=[['Evidence encoder','HistGradientBoosting'],['Max. iterations',hp['max_iter']],['Max. leaves',hp['max_leaf_nodes']],['Learning rate',hp['learning_rate']],['L2 regularization',hp['l2_regularization']],['Min. leaf samples',hp['min_samples_leaf']],['Graph context','6 train-only count/degree features'],['E-process','Online randomized ranks, kappa=0.5'],['Alarm threshold','1/alpha, alpha=0.05'],['Policy actions','observe/inspect/expand/alert/dismiss']]
    tex('table07_architecture.tex','Frozen AnytimeGraph-Guard architecture and hyperparameters.','tab:arch',['Component','Setting'],rows,resize=False,align='lp{4.2cm}')

    for no,task,caption in [(8,'binary','Binary detection macro-F1.'),(9,'multiclass','Native multiclass macro-F1; grouped macro-class in parentheses.')]:
        p=result_pivot(task); models=sorted(p)
        if task=='binary': rows=[[m,p[m].get('toniot','--'),p[m].get('edgeiiot','--'),p[m].get('apa_ddos','--')] for m in models]
        else:
            q=result_pivot('macro'); rows=[[m,f"{p[m].get('toniot','--')} ({q[m].get('toniot','--')})",f"{p[m].get('edgeiiot','--')} ({q[m].get('edgeiiot','--')})",f"{p[m].get('apa_ddos','--')} ({q[m].get('apa_ddos','--')})"] for m in models]
        tex(f'table{no:02d}_{task}.tex',caption,f'tab:{task}',['Model','ToN','Edge','APA'],rows)

    full=pd.read_csv(RESULTS/'final_test_metrics.csv'); full=full[full.task=='multiclass'].groupby(['model','dataset']).macro_f1.agg(['mean','std']).reset_index()
    ranks={};
    for ds in ['toniot','edgeiiot']:
        vals=sorted(full[full.dataset==ds]['mean'].unique(),reverse=True); ranks[ds]=(vals[0],vals[1])
    rows=[]
    for model in sorted(full.model.unique()):
        row=[esc(model)]
        for ds in ['toniot','edgeiiot','apa_ddos']:
            x=full[(full.model==model)&(full.dataset==ds)].iloc[0]; mean=f"{x['mean']:.3f}"; sd=f"{x['std']:.3f}"
            if ds in ranks and np.isclose(x['mean'],ranks[ds][0]): mean=r'\textbf{'+mean+'}'
            elif ds in ranks and np.isclose(x['mean'],ranks[ds][1]): mean=r'\underline{'+mean+'}'
            row.append(mean+r' \(\pm '+sd+r'\)')
        rows.append(row)
    tex_raw('table10_full_benchmarks.tex','Full five-seed benchmark comparison on the primary metric.','tab:fullbench',['Model','ToN-IoT','Edge-IIoT','APA-DDoS'],rows,foot='Bold and underline mark the best and second-best nonsaturated result; APA is not ranked because 15 models attain 1.000.')

    seq=pd.read_csv(RESULTS/'sequential_summary.csv')
    seq_apa=seq[seq.dataset=='apa_ddos']
    rows=[[r.method,int(r.benign_windows),f'{r.empirical_type_i:.3f}',int(r.attack_episodes),f'{r.detection_rate:.3f}',f'{r.mean_detection_delay:.1f}',f'{r.average_sample_number:.1f}'] for _,r in seq_apa.iterrows()]
    tex('table11_core_sequential.tex',r'Candidate-specific sequential stopping results on the APA-DDoS source-disjoint stream ($\alpha=0.05$).','tab:seq',['Method','Benign W','Type-I','Attacks','Detect','Delay','ASN'],rows)

    # Extension: the proposed method's anytime evidence on all three datasets,
    # not only the source-disjoint APA-DDoS control. ToN/Edge use a
    # within-group row-order proxy for time rather than a verified stream
    # (Table III), so these are reported as a weaker-ordering extension.
    seq_prop=seq[seq.method=='AnytimeGraph-Guard']
    ds_label={'apa_ddos':'APA-DDoS','toniot':'ToN-IoT','edgeiiot':'Edge-IIoT'}
    rows=[]
    for ds in ['apa_ddos','toniot','edgeiiot']:
        r=seq_prop[seq_prop.dataset==ds].iloc[0]
        rows.append([ds_label[ds],int(r.benign_windows),
                     f'{r.empirical_type_i:.3f} [{r.type_i_ci95_low:.3f},{r.type_i_ci95_high:.3f}]',
                     int(r.attack_episodes),
                     f'{r.detection_rate:.3f} [{r.detection_rate_ci95_low:.3f},{r.detection_rate_ci95_high:.3f}]',
                     f'{r.average_sample_number:.1f}', r.stream_basis])
    tex('table19_sequential_extended.tex',
        r'Proposed-method anytime evidence on all three datasets, with exact 95\% Clopper--Pearson intervals ($\alpha=0.05$).',
        'tab:seqext',['Dataset','Benign W','Type-I [95% CI]','Attacks','Detect [95% CI]','ASN','Stream basis'],
        rows, align='lccccc'+'p{2.4cm}', resize=True)

    # Drift-injection stress test of the exchangeability assumption.
    drift=pd.read_csv(RESULTS/'eprocess_drift_stress.csv')
    rows=[]
    for _,r in drift.sort_values(['dataset','shift_calibration_sd']).iterrows():
        rows.append([ds_label[r.dataset], f'{r.shift_calibration_sd:g}', int(r.benign_windows), int(r.alarms),
                     f'{r.empirical_type_i:.3f} [{r.ci95_low:.3f},{r.ci95_high:.3f}]'])
    tex('table20_drift_stress.tex',
        r'Empirical Type-I rate under synthetic benign-score drift (severity = additive shift in calibration SDs, or an absolute floor when that SD is $\approx0$; target $\alpha=0.05$).',
        'tab:drift',['Dataset','Shift','Benign W','Alarms','Type-I [95% CI]'],rows)

    # APA-DDoS device-identity confound control.
    conf=pd.read_csv(RESULTS/'apa_device_identity_confound.csv')
    rows=[[r.split, int(r.n_rows), int(r.n_devices), int(r.rows_from_devices_unseen_in_train),
           f'{r.device_identity_only_macro_f1:.3f}', f'{r.majority_class_baseline_macro_f1:.3f}']
          for _,r in conf.iterrows()]
    tex('table21_apa_confound.tex',
        'APA-DDoS device-identity-only confound control (train-only device frequency, no flow features).',
        'tab:confound',['Split','Rows','Devices','Rows: unseen device','Device-only F1','Majority-class F1'],rows,
        foot='Validation and test devices are entirely unseen in training (source-disjoint split); device-identity-only performance equals the majority-class baseline, so device fingerprinting does not explain APA saturation.')

    ab=pd.read_csv(RESULTS/'ablation_results.csv').groupby(['variant','dataset']).macro_f1.agg(['mean','std']).reset_index(); rows=[]
    for v,g in ab.groupby('variant'): rows.append([v]+[pm(g[g.dataset==d]['mean'].iloc[0],g[g.dataset==d]['std'].iloc[0]) for d in ['toniot','edgeiiot','apa_ddos']])
    tex('table12_ablation.tex','Ablation test macro-F1 (three seeds).','tab:abl',['Variant','ToN','Edge','APA'],rows)

    low=pd.read_csv(RESULTS/'low_label_results.csv').groupby(['labeled_fraction','dataset']).macro_f1.agg(['mean','std']).reset_index(); rows=[]
    for frac,g in low.groupby('labeled_fraction'): rows.append([f'{100*frac:g} percent']+[pm(g[g.dataset==d]['mean'].iloc[0],g[g.dataset==d]['std'].iloc[0]) for d in ['toniot','edgeiiot','apa_ddos']])
    tex('table13_low_label.tex','Low-label test macro-F1 (three seeds).','tab:low',['Labels','ToN','Edge','APA'],rows)

    ro=pd.read_csv(RESULTS/'robustness_results.csv').groupby(['condition','dataset']).macro_f1.agg(['mean','std']).reset_index(); rows=[]
    for c,g in ro.groupby('condition'): rows.append([c]+[pm(g[g.dataset==d]['mean'].iloc[0],g[g.dataset==d]['std'].iloc[0]) for d in ['toniot','edgeiiot','apa_ddos']])
    tex('table14_robustness.tex','Structured robustness test macro-F1 (three seeds).','tab:robust',['Condition','ToN','Edge','APA'],rows)

    f=pd.read_csv(RESULTS/'final_test_metrics.csv'); f=f[(f.model=='AnytimeGraph-Guard')&(f.task=='multiclass')]; rows=[]
    for ds,g in f.groupby('dataset'): rows.append([ds,pm(g.ece.mean(),g.ece.std()),pm(g.brier.mean(),g.brier.std()),pm(g.nll.mean(),g.nll.std())])
    tex('table15_calibration.tex','Calibrated uncertainty metrics (five seeds; lower is better).','tab:cal',['Dataset','ECE','Brier','NLL'],rows)

    rt=pd.read_csv(RESULTS/'runtime.csv').groupby('model').agg(train=('train_seconds','median'),latency=('latency_ms_per_row','median')).reset_index(); rows=[[r.model,f'{r.train:.2f}',f'{r.latency:.4f}'] for _,r in rt.iterrows()]
    tex('table16_runtime.tex','Measured desktop runtime (median across datasets and seeds).','tab:runtime',['Model','Train (s)','ms/flow'],rows,foot='Hardware-proxy timing only; no target edge-device or energy claim is made.')

    st=pd.read_csv(RESULTS/'statistical_tests.csv'); st=st[(st.scope=='dataset') & st.baseline.isin(['LightGBM','XGBoost','CatBoost','ExtraTrees','HistGradientBoosting'])]; rows=[[r.dataset,r.baseline,f'{r.mean_difference:+.3f}',f'[{r.ci95_low:+.3f},{r.ci95_high:+.3f}]',f'{r.paired_t_p_holm:.3f}',f'{r.tost_p_holm:.3f}',r.outcome] for _,r in st.iterrows()]
    tex('table17_statistics.tex','Corrected paired comparisons with strong ensembles.','tab:stats',['Dataset','Baseline','Delta F1','95 percent CI','p-Holm','p-TOST-Holm','Outcome'],rows)

    w=pd.read_csv(RESULTS/'win_tie_loss_summary.csv').iloc[0]; rows=[['Statistically superior',int(w.statistically_superior)],['Mean-superior',int(w.mean_superior)],['Statistical ties',int(w.statistical_ties)],['Practical ties',int(w.practical_ties)],['Losses',int(w.losses)],['Higher aggregate mean',f"{int(w.benchmarks_beaten_by_aggregate_mean)}/{int(w.total_real_benchmarks)}"]]
    tex('table18_wtl.tex','Aggregate benchmark win/tie/loss accounting.','tab:wtl',['Outcome','Count'],rows,resize=False)

    en=pd.read_csv(RESULTS/'enhancement_results.csv')
    stats=pd.read_csv(RESULTS/'enhancement_statistics.csv').set_index('dataset')
    rows=[]
    for ds,label in [('toniot','ToN'),('edgeiiot','Edge'),('apa_ddos','APA')]:
        g=en[en.dataset==ds]; base=g[g.model=='Frozen base'].macro_f1; dual=g[g.model=='Validation-gated dual booster'].macro_f1
        rows.append([label,pm(base.mean(),base.std()),pm(dual.mean(),dual.std()),f'{dual.mean()-base.mean():+.3f}',f"{g.selected_base_weight.mean():.2f}",f"{stats.loc[ds,'paired_t_p']:.3f}"])
    tex('paper_enhancement.tex','Exploratory validation-gated enhancement (five seeds).','tab:enhance',['Dataset','Frozen base','Dual booster','Delta','Base weight','Paired p'],rows,foot='Weights and temperatures use validation labels only. Test effects are post-hoc because the original held-out results had already been examined.')

    rt=pd.read_csv(RESULTS/'runtime.csv'); methods=['AnytimeGraph-Guard','LightGBM','XGBoost','Graph-context MLP','GATv2']; rows=[]
    for model in methods:
        row=[esc(model)]
        for ds in ['toniot','edgeiiot','apa_ddos']:
            g=rt[(rt.model==model)&(rt.dataset==ds)]; row.append(f"{g.train_seconds.mean():.2f} / {g.latency_ms_per_row.mean():.4f}")
        rows.append(row)
    tex_raw('paper_efficiency.tex','Measured training/inference cost (seconds / milliseconds per flow; five-seed means).','tab:efficiency',['Method','ToN','Edge','APA'],rows,foot='All values are software wall-clock measurements on the same development machine. They are not energy, FLOP, memory, or target-device measurements; the post-hoc dual booster was not timed.')


if __name__=='__main__':main()
