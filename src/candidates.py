"""Credible reduced prototypes for the supported predeclared candidates."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


class RelationKAN(torch.nn.Module):
    """Relation-conditioned univariate RBF-spline message functions.

    The first six inputs are graph-neighborhood statistics. Each relation learns
    a feature-wise affine grid adaptation before an additive Gaussian spline
    basis; the residual linear head sees the remaining tabular evidence.
    """
    def __init__(self, d_in: int, n_rel: int, n_classes: int, grid: int = 8):
        super().__init__()
        self.graph_dim = min(6, d_in)
        self.grid = grid
        self.rel_scale = torch.nn.Embedding(n_rel + 1, self.graph_dim)
        self.rel_shift = torch.nn.Embedding(n_rel + 1, self.graph_dim)
        torch.nn.init.ones_(self.rel_scale.weight)
        torch.nn.init.zeros_(self.rel_shift.weight)
        self.register_buffer("centers", torch.linspace(-3, 3, grid))
        self.spline = torch.nn.Linear(self.graph_dim * grid, n_classes, bias=False)
        self.residual = torch.nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        g = x[:, -self.graph_dim:]
        z = g * self.rel_scale(relation) + self.rel_shift(relation)
        basis = torch.exp(-0.5 * ((z.unsqueeze(-1) - self.centers) / 0.8) ** 2)
        return self.spline(basis.flatten(1)) + self.residual(x)

    def spline_regularization(self) -> torch.Tensor:
        w = self.spline.weight.view(self.spline.out_features, self.graph_dim, self.grid)
        return (w[:, :, 1:] - w[:, :, :-1]).abs().mean()


class ParameterMatchedMLP(torch.nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, n_classes))
    def forward(self, x: torch.Tensor, relation: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x)


def train_torch(model, x, y, relation, seed, epochs=12, lr=3e-3, reg=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.as_tensor(x, dtype=torch.float32); yt = torch.as_tensor(y, dtype=torch.long)
    rt = torch.as_tensor(relation, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt, yt, rt), batch_size=512, shuffle=True, generator=gen)
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for xb, yb, rb in loader:
            xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
            loss = torch.nn.functional.cross_entropy(model(xb, rb), yb)
            if reg and hasattr(model, "spline_regularization"):
                loss = loss + reg * model.spline_regularization()
            opt.zero_grad(); loss.backward(); opt.step()
    return model, time.perf_counter() - start


@torch.no_grad()
def torch_proba(model, x, relation):
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(x), 4096):
        xb = torch.as_tensor(x[start:start+4096], dtype=torch.float32, device=device)
        rb = torch.as_tensor(relation[start:start+4096], dtype=torch.long, device=device)
        out.append(torch.softmax(model(xb, rb), dim=1).cpu().numpy())
    return np.row_stack(out)


class HDGraphClassifier:
    """Bipolar hyperdimensional graph classifier with online prototypes."""
    def __init__(self, dim=2048, seed=0): self.dim, self.seed = dim, seed
    def fit(self, x, y):
        rng = np.random.default_rng(self.seed)
        self.proj = rng.choice(np.array([-1, 1], np.int8), size=(x.shape[1], self.dim))
        self.classes_ = np.unique(y)
        self.proto = np.zeros((len(self.classes_), self.dim), np.float32)
        for start in range(0, len(x), 1024):
            h = np.where(x[start:start+1024] @ self.proj >= 0, 1, -1).astype(np.int8)
            for c in self.classes_:
                self.proto[c] += h[y[start:start+1024] == c].sum(axis=0)
        self.proto = np.where(self.proto >= 0, 1, -1).astype(np.int8)
        return self
    def predict_proba(self, x):
        scores=[]
        for start in range(0,len(x),1024):
            h=np.where(x[start:start+1024] @ self.proj >= 0,1,-1).astype(np.int8)
            scores.append((h.astype(np.float32) @ self.proto.T.astype(np.float32))/self.dim)
        z=np.vstack(scores); z=z-z.max(1,keepdims=True); e=np.exp(z*8)
        return e/e.sum(1,keepdims=True)


def anytime_e_process(attack_probability: np.ndarray, calibration_benign: np.ndarray, alpha=0.05, betting=0.5):
    """Conformal-rank e-values accumulated with a bounded betting process."""
    cal = np.sort(np.asarray(calibration_benign))
    # Super-uniform upper-tail conformal p-values under exchangeability.
    p = (len(cal) - np.searchsorted(cal, attack_probability, side="left") + 1) / (len(cal) + 1)
    e = betting * np.power(np.clip(p, 1e-12, 1), betting - 1)
    process = np.cumprod(e)
    alarm = np.flatnonzero(process >= 1 / alpha)
    return {"p_values": p, "e_values": e, "process": process, "alarm_index": int(alarm[0]) if len(alarm) else None}


def condensed_coreset(x, y, per_class=32, seed=0):
    centers=[]; labels=[]
    for c in np.unique(y):
        xc=x[y==c]; k=min(per_class,len(xc))
        km=MiniBatchKMeans(k,random_state=seed,batch_size=1024,n_init=1).fit(xc)
        centers.append(km.cluster_centers_); labels.extend([c]*k)
    return np.row_stack(centers),np.asarray(labels)


class ParticleBelief:
    """Particle approximation over attack-class belief under missing telemetry."""
    def __init__(self, n_particles=256, seed=0): self.n_particles=n_particles; self.rng=np.random.default_rng(seed)
    def filter(self, likelihoods):
        k=likelihoods.shape[1]; particles=self.rng.integers(0,k,self.n_particles); w=np.full(self.n_particles,1/self.n_particles)
        beliefs=[]; ess=[]
        for likelihood in likelihoods:
            w*=np.clip(likelihood[particles],1e-8,1); w/=w.sum()
            beliefs.append(np.bincount(particles,weights=w,minlength=k))
            ess.append(1/np.square(w).sum())
            if ess[-1] < self.n_particles/2:
                particles=self.rng.choice(particles,size=self.n_particles,p=w); w.fill(1/self.n_particles)
            # proposal allows hypotheses to change slowly
            change=self.rng.random(self.n_particles)<0.03; particles[change]=self.rng.integers(0,k,change.sum())
        return np.asarray(beliefs),np.asarray(ess)


def structured_graph_shift(x, feature_idx=None, topology_idx=None, protocol_idx=None,
                           radius=0.1, seed=0):
    """Semantics-scoped perturbation used by the partial Wasserstein prototype.

    It perturbs only declared feasible feature, topology, and protocol groups;
    callers must not infer absent physical semantics from column positions.
    """
    rng=np.random.default_rng(seed); out=np.asarray(x,dtype=np.float32).copy()
    feature_idx=list(feature_idx or []); topology_idx=list(topology_idx or []); protocol_idx=list(protocol_idx or [])
    if feature_idx:
        noise=rng.normal(size=(len(out),len(feature_idx))).astype(np.float32)
        norm=np.linalg.norm(noise,axis=1,keepdims=True)+1e-8
        out[:,feature_idx]+=radius*noise/norm
    if topology_idx:
        mask=rng.random((len(out),len(topology_idx)))<min(.5,radius)
        vals=out[:,topology_idx]; vals[mask]=0; out[:,topology_idx]=vals
    if protocol_idx:
        take=rng.random(len(out))<min(.5,radius)
        out[np.ix_(take,protocol_idx)]=0
    return out


def hardware_proxy_search(builders, sample, repeats=5):
    """Measured desktop Pareto screen; intentionally not an energy/device claim."""
    import pickle
    rows=[]
    for name,model in builders.items():
        timings=[]
        for _ in range(repeats):
            start=time.perf_counter()
            if hasattr(model,'predict_proba'): model.predict_proba(sample)
            else: model(sample)
            timings.append(time.perf_counter()-start)
        rows.append({'architecture':name,'latency_ms':1000*np.median(timings),
                     'serialized_bytes':len(pickle.dumps(model))})
    for row in rows:
        row['pareto']=not any((z['latency_ms']<=row['latency_ms'] and z['serialized_bytes']<=row['serialized_bytes'] and
                               (z['latency_ms']<row['latency_ms'] or z['serialized_bytes']<row['serialized_bytes'])) for z in rows)
    return rows


def fit_group_dro_binary(x, y, environment, seed=0, epochs=100, eta=0.05):
    """Small multi-environment GroupDRO classifier for candidate 10 smoke tests."""
    torch.manual_seed(seed); xt=torch.as_tensor(x,dtype=torch.float32); yt=torch.as_tensor(y,dtype=torch.long)
    et=torch.as_tensor(environment,dtype=torch.long); envs=torch.unique(et)
    model=torch.nn.Linear(x.shape[1],2); opt=torch.optim.AdamW(model.parameters(),lr=1e-2,weight_decay=1e-4)
    q=torch.ones(len(envs))/len(envs)
    for _ in range(epochs):
        losses=[]
        for e in envs:
            mask=et==e; losses.append(torch.nn.functional.cross_entropy(model(xt[mask]),yt[mask]))
        lv=torch.stack(losses); q=q*torch.exp(eta*lv.detach()); q=q/q.sum()
        loss=(q*lv).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return model,q.detach().numpy()
