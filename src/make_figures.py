"""Generate publication figures directly from saved result files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ARTIFACTS, FIGURES, RESULTS

mpl.rcParams.update({'font.family':'serif','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42})
sns.set_theme(style='whitegrid',context='paper',font_scale=.85)


def save(fig,name):
    fig.savefig(FIGURES/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIGURES/f'{name}.png',dpi=300,bbox_inches='tight')
    plt.close(fig)


def boxes(name,items,arrows):
    fig,ax=plt.subplots(figsize=(7.16,2.25)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    for x,y,w,h,text,color in items:
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012,rounding_size=.015',fc=color,ec='#23395d',lw=1)
        ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7)
    for a,b in arrows:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color='#34495e',lw=1))
    save(fig,name)


def main():
    inv=json.load(open(ARTIFACTS/'audit'/'dataset_inventory.json',encoding='utf8'))
    total=len(inv['datasets'])
    boxes('fig01_dataset_screening',[
      (.02,.36,.15,.28,f'{total} local\ncorpora\n112.2M rows','#dceaf7'),(.22,.36,.16,.28,'Schema, label,\nduplicate and\nidentifier audit','#e8f1de'),(.43,.57,.16,.24,'Exclude: no IDS\nlabels/graph/time','#f9e3e3'),(.43,.16,.16,.24,'Feasible graph or\nordered candidate\nassumptions','#e4f2e6'),(.66,.36,.14,.28,'3 frozen\nprimary\ncorpora','#fff1cc'),(.85,.36,.13,.28,'Validation-only\ncandidate\nselection','#eadff2')],
      [((.17,.5),(.22,.5)),((.38,.5),(.43,.69)),((.38,.5),(.43,.28)),((.59,.28),(.66,.5)),((.80,.5),(.85,.5))])
    boxes('fig02_leakage_pipeline',[
      (.02,.36,.14,.28,'Raw CSV +\nrow identity','#dceaf7'),(.20,.36,.14,.28,'Deduplicate;\nconflict audit','#e8f1de'),(.38,.36,.14,.28,'Group folds\n3/1/1','#fff1cc'),(.56,.62,.17,.22,'Train: fit maps,\nimputer, encoder','#e4f2e6'),(.56,.10,.17,.22,'Validation: tune,\ncalibrate, select','#eadff2'),(.79,.36,.18,.28,'One-time test\nafter frozen\nselection','#f9e3e3')],
      [((.16,.5),(.20,.5)),((.34,.5),(.38,.5)),((.52,.5),(.56,.73)),((.52,.5),(.56,.21)),((.73,.73),(.79,.55)),((.73,.21),(.79,.45))])

    rank=pd.read_csv(RESULTS/'top3_tuned_validation_summary.csv').sort_values('macro_f1_median')
    fig,ax=plt.subplots(figsize=(3.5,2.25)); ax.barh(rank.candidate,rank.macro_f1_median,xerr=rank.macro_f1_std,color=['#9ecae1','#fdae6b','#74c476']); ax.scatter(rank.macro_f1_worst,rank.candidate,marker='|',s=90,color='black',label='worst seed'); ax.set(xlabel='Validation macro-F1',xlim=(0,1.02)); ax.legend(frameon=False); save(fig,'fig03_candidate_ranking')

    boxes('fig04_architecture',[
      (.02,.58,.14,.22,'Flow/event\nfeatures','#dceaf7'),(.02,.18,.14,.22,'Endpoint and\nrelation IDs','#dceaf7'),(.22,.36,.17,.28,'Train-only graph\ncount/degree\ncontext','#e8f1de'),(.45,.36,.16,.28,'Calibrated\ngraph evidence\nencoder','#fff1cc'),(.67,.58,.14,.22,'Online-rank\ne-values','#eadff2'),(.67,.18,.14,.22,'Cost-sensitive\ntelemetry policy','#eadff2'),(.86,.36,.12,.28,'Gated alert\nor dismiss','#f9e3e3')],
      [((.16,.69),(.22,.55)),((.16,.29),(.22,.45)),((.39,.5),(.45,.5)),((.61,.5),(.67,.69)),((.61,.5),(.67,.29)),((.81,.69),(.86,.55)),((.81,.29),(.86,.45))])

    f=pd.read_csv(RESULTS/'final_test_metrics.csv'); m=f[f.task=='multiclass'].groupby(['dataset','model']).macro_f1.mean().reset_index()
    keep=['AnytimeGraph-Guard','LightGBM','XGBoost','ExtraTrees','HistGradientBoosting','MLP']
    fig,ax=plt.subplots(figsize=(7.16,2.55)); sns.barplot(data=m[m.model.isin(keep)],x='dataset',y='macro_f1',hue='model',ax=ax); ax.set(xlabel='',ylabel='Test macro-F1',ylim=(0.7,1.01)); ax.legend(ncol=3,frameon=False,loc='lower right'); save(fig,'fig05_performance')

    pc=pd.read_csv(RESULTS/'per_class_recall.csv'); pc=pc[pc.model=='AnytimeGraph-Guard'].groupby(['dataset','class']).recall.mean().reset_index(); pc['key']=pc.dataset+': '+pc['class'];
    mat=pc.set_index('key')[['recall']]
    fig,ax=plt.subplots(figsize=(3.5,6.0)); sns.heatmap(mat,cmap='YlGnBu',vmin=0,vmax=1,annot=True,fmt='.2f',cbar_kws={'label':'Recall'},ax=ax); ax.set(xlabel='',ylabel=''); save(fig,'fig06_per_class_recall')

    cm=pd.read_csv(RESULTS/'confusion_matrices.csv'); cm=cm[cm.model=='AnytimeGraph-Guard'].groupby(['dataset','true','predicted']).value.mean().reset_index()
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.35))
    for ax,(ds,g) in zip(axes,cm.groupby('dataset')):
        z=g.pivot(index='true',columns='predicted',values='value').fillna(0); sns.heatmap(z,cmap='Blues',vmin=0,vmax=1,cbar=False,xticklabels=False,yticklabels=False,ax=ax); ax.set_title(ds); ax.set(xlabel='Predicted',ylabel='True')
    save(fig,'fig07_confusions')

    ep=pd.read_csv(RESULTS/'eprocess_trajectories.csv'); ep=ep[ep.dataset=='apa_ddos']; seed=ep.seed.min(); sub=ep[ep.seed==seed]
    fig,ax=plt.subplots(figsize=(3.5,2.35))
    for (grp,y),g in sub.groupby(['group','true_attack']): ax.plot(g.t,g.log_e,label=('attack' if y else 'benign')+f' {str(grp)[-3:]}',alpha=.85)
    ax.axhline(np.log(20),ls='--',color='black',label=r'$\log(1/\alpha)$'); ax.set(xlabel='Observed flows',ylabel='Log e-process',xlim=(0,500)); ax.legend(frameon=False,ncol=2); save(fig,'fig08_eprocess')

    ab=pd.read_csv(RESULTS/'ablation_results.csv').groupby(['dataset','variant']).macro_f1.mean().reset_index();
    fig,ax=plt.subplots(figsize=(7.16,2.5)); sns.barplot(data=ab,x='dataset',y='macro_f1',hue='variant',ax=ax); ax.set(xlabel='',ylabel='Macro-F1',ylim=(0,1.02)); ax.legend(ncol=3,frameon=False); save(fig,'fig09_ablation')

    low=pd.read_csv(RESULTS/'low_label_results.csv').groupby(['dataset','labeled_fraction']).macro_f1.agg(['mean','std']).reset_index()
    fig,ax=plt.subplots(figsize=(3.5,2.4));
    for ds,g in low.groupby('dataset'): ax.plot(g.labeled_fraction*100,g['mean'],marker='o',label=ds); ax.fill_between(g.labeled_fraction*100,g['mean']-g['std'].fillna(0),g['mean']+g['std'].fillna(0),alpha=.12)
    ax.set(xlabel='Labeled training data (%)',ylabel='Test macro-F1',xscale='log',ylim=(0.5,1.02)); ax.legend(frameon=False); save(fig,'fig10_low_label')

    ro=pd.read_csv(RESULTS/'enhanced_robustness_results.csv')
    ro['setting']=ro.apply(lambda r: 'clean' if r['family']=='clean' else ('topology removed' if r['family']=='topology' else f"{r['condition']} {r['severity']:.2f}"),axis=1)
    order=['clean','missing features 0.10','missing features 0.20','missing features 0.30','feature noise 0.05','feature noise 0.10','feature noise 0.20','graph-context corruption 0.10','graph-context corruption 0.20','graph-context corruption 0.40','topology removed']
    ro=ro.groupby(['dataset','setting']).macro_f1.mean().reset_index().pivot(index='setting',columns='dataset',values='macro_f1').reindex(order)
    ro=ro[['toniot','edgeiiot','apa_ddos']]; ro.columns=['ToN','Edge','APA']
    fig,ax=plt.subplots(figsize=(3.5,3.2)); sns.heatmap(ro,cmap='YlOrRd_r',vmin=0,vmax=1,annot=True,fmt='.2f',annot_kws={'fontsize':6.5},ax=ax,cbar_kws={'label':'Macro-F1'}); ax.set(xlabel='',ylabel=''); ax.tick_params(axis='y',labelsize=6.5); save(fig,'fig11_robustness')

    cal=pd.read_csv(RESULTS/'reliability_bins.csv'); cal=cal.groupby(['dataset','bin_low','bin_high']).apply(lambda g:pd.Series({'conf':np.average(g.mean_confidence,weights=g['count']),'acc':np.average(g.accuracy,weights=g['count'])}),include_groups=False).reset_index()
    fig,ax=plt.subplots(figsize=(3.5,2.5)); ax.plot([0,1],[0,1],'--',color='gray')
    for ds,g in cal.groupby('dataset'): ax.plot(g.conf,g.acc,marker='o',label=ds)
    ax.set(xlabel='Mean confidence',ylabel='Empirical accuracy',xlim=(0,1),ylim=(0,1)); ax.legend(frameon=False); save(fig,'fig12_calibration')

    rt=pd.read_csv(RESULTS/'runtime.csv').groupby(['dataset','model']).latency_ms_per_row.median().reset_index(); perf=m.merge(rt,on=['dataset','model'])
    fig,ax=plt.subplots(figsize=(3.5,2.5));
    for ds,g in perf.groupby('dataset'): ax.scatter(g.latency_ms_per_row,g.macro_f1,label=ds,s=20,alpha=.8)
    prop=perf[perf.model=='AnytimeGraph-Guard']; ax.scatter(prop.latency_ms_per_row,prop.macro_f1,marker='*',s=90,color='black',label='proposed'); ax.set(xlabel='Latency (ms/flow)',ylabel='Macro-F1',xscale='log'); ax.legend(frameon=False); save(fig,'fig13_runtime_performance')

    wt=pd.read_csv(RESULTS/'win_tie_loss_summary.csv').iloc[0]; vals=[wt.statistically_superior,wt.mean_superior,wt.statistical_ties+wt.practical_ties,wt.losses]
    fig,ax=plt.subplots(figsize=(3.5,2.15)); ax.bar(['Stat.\nsuperior','Mean-\nsuperior','Ties','Losses'],vals,color=['#2ca25f','#99d8c9','#bdbdbd','#de2d26']);
    for i,v in enumerate(vals): ax.text(i,v+.15,str(int(v)),ha='center'); ax.set(ylabel='Benchmarks',ylim=(0,max(vals)+1.5)); save(fig,'fig14_win_tie_loss')

    # Post-hoc enhancement: values are validation-gated and test results are exploratory.
    en=pd.read_csv(RESULTS/'enhancement_results.csv')
    ens=en.groupby(['dataset','model']).macro_f1.agg(['mean','std']).reset_index()
    fig,ax=plt.subplots(figsize=(3.5,2.35))
    order=['toniot','edgeiiot','apa_ddos']; x=np.arange(3); width=.35
    base=ens[ens.model=='Frozen base'].set_index('dataset').reindex(order)
    dual=ens[ens.model=='Validation-gated dual booster'].set_index('dataset').reindex(order)
    ax.bar(x-width/2,base['mean'],width,yerr=base['std'],label='Frozen base',color='#9ecae1',capsize=2)
    ax.bar(x+width/2,dual['mean'],width,yerr=dual['std'],label='Dual booster',color='#31a354',capsize=2)
    for i,(a,b) in enumerate(zip(base['mean'],dual['mean'])): ax.text(i,b+.012,f'{b-a:+.3f}',ha='center',fontsize=7)
    ax.set(xticks=x,xticklabels=['ToN','Edge','APA'],ylabel='Exploratory test macro-F1',ylim=(.78,1.025)); ax.legend(frameon=False,loc='lower right'); save(fig,'fig15_enhancement')

    # Annotated row-normalized confusion matrices.  Cell numbers are percentages.
    cm=pd.read_csv(RESULTS/'enhanced_confusion_matrices.csv').groupby(['dataset','true','predicted']).row_fraction.mean().reset_index()
    short={'DDoS_HTTP':'DDoS-H','DDoS_ICMP':'DDoS-I','DDoS_TCP':'DDoS-T','DDoS_UDP':'DDoS-U','Fingerprinting':'Finger','Port_Scanning':'PortScan','Ransomware':'Ransom','SQL_injection':'SQL-inj','Vulnerability_scanner':'VulnScan','DDoS-PSH-ACK':'PSH-ACK','DDoS-ACK':'ACK'}
    fig=plt.figure(figsize=(7.16,6.05)); gs=fig.add_gridspec(2,2,height_ratios=[1.25,1],hspace=.43,wspace=.34)
    axes={'edgeiiot':fig.add_subplot(gs[0,:]),'toniot':fig.add_subplot(gs[1,0]),'apa_ddos':fig.add_subplot(gs[1,1])}
    for ds,ax in axes.items():
        g=cm[cm.dataset==ds]; labels=g.true.drop_duplicates().tolist(); z=g.pivot(index='true',columns='predicted',values='row_fraction').reindex(index=labels,columns=labels).fillna(0)
        ticks=[short.get(v,v) for v in labels]
        sns.heatmap(z*100,cmap='Blues',vmin=0,vmax=100,annot=True,fmt='.0f',annot_kws={'fontsize':5.2},cbar=ds=='edgeiiot',cbar_kws={'label':'Row percent'},xticklabels=ticks,yticklabels=ticks,ax=ax,square=True)
        ax.set_title({'edgeiiot':'Edge-IIoT','toniot':'ToN-IoT','apa_ddos':'APA-DDoS'}[ds]); ax.set(xlabel='Predicted class',ylabel='True class'); ax.tick_params(axis='x',rotation=55,labelsize=5.5); ax.tick_params(axis='y',rotation=0,labelsize=5.5)
    save(fig,'fig16_annotated_confusions')

    # Validation-only one-factor-at-a-time sensitivity.
    se=pd.read_csv(RESULTS/'sensitivity_results.csv')
    factors=['base_blend_weight','max_iter','max_leaf_nodes','learning_rate','min_samples_leaf']
    titles=['Base blend weight','Iterations','Maximum leaves','Learning rate','Minimum leaf size']
    fig,axes=plt.subplots(2,3,figsize=(7.16,4.2)); axes=axes.ravel()
    for ax,factor,title in zip(axes,factors,titles):
        g=se[se.factor==factor].groupby(['dataset','value']).validation_macro_f1.agg(['mean','std']).reset_index()
        for ds,h in g.groupby('dataset'):
            ax.plot(h.value,h['mean'],marker='o',label={'toniot':'ToN','edgeiiot':'Edge','apa_ddos':'APA'}[ds]); ax.fill_between(h.value,h['mean']-h['std'].fillna(0),h['mean']+h['std'].fillna(0),alpha=.10)
        ax.set_title(title); ax.set_ylim(.55,1.02); ax.set_ylabel('Validation macro-F1'); ax.grid(alpha=.25)
    axes[5].axis('off'); handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,ncol=3,frameon=False,loc='lower right',bbox_to_anchor=(.96,.055)); fig.subplots_adjust(hspace=.52,wspace=.30,bottom=.16); save(fig,'fig17_sensitivity')

    # Training/validation loss and accuracy convergence for all three datasets.
    cv=pd.read_csv(RESULTS/'convergence_curves.csv')
    fig,axes=plt.subplots(3,2,figsize=(7.16,5.0),sharex=False)
    for row,ds in enumerate(['toniot','edgeiiot','apa_ddos']):
        d=cv[cv.dataset==ds]; max_common=d.groupby('seed').iteration.max().min(); d=d[d.iteration<=max_common]
        for col,metric in enumerate(['loss','accuracy']):
            ax=axes[row,col]
            for split,color in [('training','#3182bd'),('validation','#e6550d')]:
                g=d[d.split==split].groupby('iteration')[metric].agg(['mean','std']).reset_index(); ax.plot(g.iteration,g['mean'],label=split,color=color); ax.fill_between(g.iteration,g['mean']-g['std'].fillna(0),g['mean']+g['std'].fillna(0),color=color,alpha=.12)
            ax.set_ylabel(f"{ {'toniot':'ToN','edgeiiot':'Edge','apa_ddos':'APA'}[ds] } {metric}")
            ax.set_xlabel('Boosting iteration'); ax.grid(alpha=.25)
            if metric=='accuracy': ax.set_ylim(0,1.02)
    axes[0,0].legend(frameon=False,ncol=2); save(fig,'fig18_convergence')

    # Multiclass one-vs-rest PR evidence plus calibration. All values are saved
    # seed-level outputs; no smoothing beyond a common recall grid is applied.
    pr=pd.read_csv(RESULTS/'pr_curve_results.csv'); pr=pr[pr.task=='macro_ovr']
    aps=pd.read_csv(RESULTS/'pr_curve_summary.csv'); aps=aps[aps.task=='macro_ovr'].groupby(['dataset','model']).average_precision.mean()
    fig,axes=plt.subplots(4,1,figsize=(3.5,6.0)); axes=axes.ravel()
    model_style={'AnytimeGraph-Guard':('#1b7837','-'),'LightGBM':('#2166ac','--'),'XGBoost':('#b2182b',':')}
    for ax,ds,title in zip(axes[:3],['toniot','edgeiiot','apa_ddos'],['ToN-IoT','Edge-IIoT','APA-DDoS']):
        d=pr[pr.dataset==ds]
        for model,(color,style) in model_style.items():
            g=d[d.model==model].groupby('recall').precision.agg(['mean','std']).reset_index(); label=('Proposed' if model=='AnytimeGraph-Guard' else model)+f" ({aps.loc[(ds,model)]:.3f})"
            ax.plot(g.recall,g['mean'],color=color,ls=style,label=label); ax.fill_between(g.recall,(g['mean']-g['std'].fillna(0)).clip(0,1),(g['mean']+g['std'].fillna(0)).clip(0,1),color=color,alpha=.09)
        ax.set(title=title,xlabel='Recall',ylabel='Macro precision',xlim=(0,1),ylim=(0,1.02)); ax.legend(frameon=False,fontsize=5.5,loc='lower left',ncol=3,columnspacing=.7,handlelength=2.0); ax.grid(alpha=.25)
    cal=pd.read_csv(RESULTS/'reliability_bins.csv'); cal=cal.groupby(['dataset','bin_low','bin_high']).apply(lambda g:pd.Series({'conf':np.average(g.mean_confidence,weights=g['count']),'acc':np.average(g.accuracy,weights=g['count'])}),include_groups=False).reset_index()
    ax=axes[3]; ax.plot([0,1],[0,1],'--',color='gray',label='ideal')
    for ds,marker in [('toniot','o'),('edgeiiot','s'),('apa_ddos','^')]:
        g=cal[cal.dataset==ds]; ax.plot(g.conf,g.acc,marker=marker,ms=3,label={'toniot':'ToN','edgeiiot':'Edge','apa_ddos':'APA'}[ds])
    ax.set(title='Frozen-base reliability',xlabel='Mean confidence',ylabel='Empirical accuracy',xlim=(0,1),ylim=(0,1.02)); ax.legend(frameon=False,fontsize=5.5,ncol=4,loc='upper left'); ax.grid(alpha=.25)
    fig.subplots_adjust(hspace=.62); save(fig,'fig19_pr_calibration')


if __name__=='__main__':main()
