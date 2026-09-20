"""Frozen inductive GraphSAGE, R-GCN, and GATv2 edge-classification baselines."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, recall_score
from torch_geometric.nn import GATv2Conv, RGCNConv, SAGEConv

from .config import ARTIFACTS, PRIMARY_FILES, RESULTS, SEEDS
from .data import load_bundle
from .final_evaluation import MACRO_MAPS, aggregate_task, stratified_cap
from .metrics import classification_metrics
from .prepare_protocol import SPECS


class EdgeGNN(torch.nn.Module):
    def __init__(self,kind,node_dim,base_dim,n_classes,n_rel):
        super().__init__(); self.kind=kind
        if kind=='GraphSAGE': self.c1=SAGEConv(node_dim,24); self.c2=SAGEConv(24,24)
        elif kind=='R-GCN': self.c1=RGCNConv(node_dim,24,n_rel*2); self.c2=RGCNConv(24,24,n_rel*2)
        else: self.c1=GATv2Conv(node_dim,12,heads=2); self.c2=GATv2Conv(24,24,heads=1)
        self.head=torch.nn.Sequential(torch.nn.Linear(48+base_dim,64),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(64,n_classes))
    def encode(self,x,edge_index,edge_type):
        if self.kind=='R-GCN': z=torch.relu(self.c1(x,edge_index,edge_type)); return torch.relu(self.c2(z,edge_index,edge_type))
        z=torch.relu(self.c1(x,edge_index)); return torch.relu(self.c2(z,edge_index))
    def decode(self,z,src,dst,base): return self.head(torch.cat([z[src],z[dst],base],1))


def prepare_graph(name,b):
    spec=SPECS[name]; raw=pd.read_csv(PRIMARY_FILES[name],low_memory=False); raw.columns=[c.strip() for c in raw.columns]
    man=pd.read_csv(ARTIFACTS/'manifests'/f'{name}_split_manifest.csv')
    frames={};
    for s in ['train','validation','test']:
        m=man[man.split==s]; frames[s]=raw.iloc[m.original_row.to_numpy()].reset_index(drop=True)
    src=frames['train'][spec['src']].fillna('<NA>').astype(str); dst=frames['train'][spec['dst']].fillna('<NA>').astype(str)
    nodes=sorted(set(src)|set(dst)); node_map={v:i+1 for i,v in enumerate(nodes)}
    rel_col=next((c for c in spec['relation'] if c in frames['train']),None)
    train_rel=frames['train'][rel_col].fillna('<NA>').astype(str) if rel_col else pd.Series('default',index=frames['train'].index)
    rel_map={v:i for i,v in enumerate(sorted(train_rel.unique()))}; nr=max(1,len(rel_map))
    outc=src.value_counts(); inc=dst.value_counts(); ou=pd.DataFrame({'s':src,'d':dst}).groupby('s').d.nunique(); iu=pd.DataFrame({'s':src,'d':dst}).groupby('d').s.nunique()
    nx=np.zeros((len(nodes)+1,4),np.float32)
    for v,i in node_map.items(): nx[i]=np.log1p([outc.get(v,0),inc.get(v,0),ou.get(v,0),iu.get(v,0)])
    endpoints={}; relations={}
    for s,f in frames.items():
        si=f[spec['src']].fillna('<NA>').astype(str).map(node_map).fillna(0).to_numpy(np.int64)
        di=f[spec['dst']].fillna('<NA>').astype(str).map(node_map).fillna(0).to_numpy(np.int64)
        rr=f[rel_col].fillna('<NA>').astype(str).map(rel_map).fillna(0).to_numpy(np.int64) if rel_col else np.zeros(len(f),np.int64)
        endpoints[s]=(si,di); relations[s]=rr
    si,di=endpoints['train']; rr=relations['train']
    edge_index=np.column_stack([np.concatenate([si,di]),np.concatenate([di,si])]).T
    edge_type=np.concatenate([rr,rr+nr])
    return nx,edge_index,edge_type,endpoints,nr


def main():
    cfg={'models':['GraphSAGE','R-GCN','GATv2'],'hidden':24,'decoder':64,'epochs':14,'training_cap':60000,'graph':'training endpoints/edges only; unseen nodes map to isolated unknown','created_before_graph_test_load':True}
    (ARTIFACTS/'protocol'/'graph_baseline_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    metrics=[]; perclass=[]; cms=[]; times=[]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name in ['toniot','edgeiiot','apa_ddos']:
      b=load_bundle(name); nx,ei,et,endpoints,nr=prepare_graph(name,b); nc=len(b.classes)
      benign=b.classes.index({'toniot':'normal','edgeiiot':'Normal','apa_ddos':'Benign'}[name])
      node_x=torch.tensor(nx,device=device); edge_index=torch.tensor(ei,dtype=torch.long,device=device); edge_type=torch.tensor(et,dtype=torch.long,device=device)
      for seed in SEEDS:
       torch.manual_seed(seed); np.random.seed(seed); idx=stratified_cap(b.y['train'],60000,seed)
       tr_src=torch.tensor(endpoints['train'][0][idx],dtype=torch.long,device=device); tr_dst=torch.tensor(endpoints['train'][1][idx],dtype=torch.long,device=device)
       tr_base=torch.tensor(b.X_base['train'][idx],device=device); tr_y=torch.tensor(b.y['train'][idx],dtype=torch.long,device=device)
       counts=np.bincount(b.y['train'][idx],minlength=nc); weight=torch.tensor(len(idx)/(nc*np.maximum(counts,1)),dtype=torch.float32,device=device)
       for kind in cfg['models']:
        model=EdgeGNN(kind,nx.shape[1],b.X_base['train'].shape[1],nc,nr).to(device); opt=torch.optim.AdamW(model.parameters(),lr=.004,weight_decay=1e-4)
        start=time.perf_counter()
        for _ in range(cfg['epochs']):
            model.train(); z=model.encode(node_x,edge_index,edge_type); logits=model.decode(z,tr_src,tr_dst,tr_base); loss=torch.nn.functional.cross_entropy(logits,tr_y,weight=weight); opt.zero_grad(); loss.backward(); opt.step()
        train_s=time.perf_counter()-start; model.eval(); start=time.perf_counter()
        with torch.no_grad():
            z=model.encode(node_x,edge_index,edge_type); ts,td=endpoints['test']; out=[]
            for j in range(0,len(ts),4096):
                xb=torch.tensor(b.X_base['test'][j:j+4096],device=device); logits=model.decode(z,torch.tensor(ts[j:j+4096],device=device),torch.tensor(td[j:j+4096],device=device),xb); out.append(torch.softmax(logits,1).cpu().numpy())
        p=np.row_stack(out); infer=time.perf_counter()-start
        for task in ['multiclass','binary','macro']:
            if task=='multiclass': yy,pp,names=b.y['test'],p,b.classes
            elif task=='binary': attack=1-p[:,benign]; yy=b.y_binary['test']; pp=np.column_stack([1-attack,attack]); names=['benign','attack']
            else: yy,pp,names=aggregate_task(b.y['test'],p,b.classes,MACRO_MAPS[name])
            metrics.append({'dataset':name,'model':kind,'seed':seed,'task':task,**classification_metrics(yy,pp),'temperature':1.0})
            if task=='multiclass':
                pred=pp.argmax(1); rec=recall_score(yy,pred,labels=np.arange(len(names)),average=None,zero_division=0)
                for c,v in zip(names,rec):perclass.append({'dataset':name,'model':kind,'seed':seed,'class':c,'recall':v})
                cm=confusion_matrix(yy,pred,labels=np.arange(len(names)),normalize='true')
                for i,a in enumerate(names):
                    for j,zv in enumerate(names):cms.append({'dataset':name,'model':kind,'seed':seed,'true':a,'predicted':zv,'value':cm[i,j]})
        times.append({'dataset':name,'model':kind,'seed':seed,'train_seconds':train_s,'validation_predict_seconds':0,'test_inference_seconds':infer,'test_rows':len(b.y['test']),'latency_ms_per_row':1000*infer/len(b.y['test'])})
    # Append atomically after all graph runs complete.
    for path,new in [('final_test_metrics.csv',pd.DataFrame(metrics)),('per_class_recall.csv',pd.DataFrame(perclass)),('confusion_matrices.csv',pd.DataFrame(cms)),('runtime.csv',pd.DataFrame(times))]:
        old=pd.read_csv(RESULTS/path); old=old[~old.model.isin(cfg['models'])]; pd.concat([old,new],ignore_index=True).to_csv(RESULTS/path,index=False)


if __name__=='__main__':main()
