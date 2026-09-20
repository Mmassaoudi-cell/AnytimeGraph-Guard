import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.candidates import (HDGraphClassifier, ParticleBelief, RelationKAN,
                            fit_group_dro_binary, structured_graph_shift)
from src.config import ARTIFACTS
from src.enhancement_study import mix
from src.sequential_evaluation import online_e


class CoreTests(unittest.TestCase):
    def test_frozen_protocol_and_group_disjoint(self):
        self.assertTrue((ARTIFACTS/'protocol'/'frozen_protocol.json').exists())
        for name in ['toniot','edgeiiot','apa_ddos']:
            m=pd.read_csv(ARTIFACTS/'manifests'/f'{name}_split_manifest.csv')
            sets={s:set(m.loc[m.split==s,'group_id'].astype(str)) for s in ['train','validation','test']}
            self.assertFalse(sets['train']&sets['validation'])
            self.assertFalse(sets['train']&sets['test'])
            self.assertFalse(sets['validation']&sets['test'])

    def test_relation_kan_shapes_and_regularizer(self):
        model=RelationKAN(12,4,3,grid=6); x=torch.randn(7,12); r=torch.arange(7)%5
        self.assertEqual(tuple(model(x,r).shape),(7,3))
        self.assertTrue(torch.isfinite(model.spline_regularization()))

    def test_hdc_probabilities(self):
        rng=np.random.default_rng(1); x=rng.normal(size=(40,5)); y=np.repeat([0,1],20)
        p=HDGraphClassifier(128,1).fit(x,y).predict_proba(x[:4])
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6)

    def test_eprocess_finite(self):
        rng=np.random.default_rng(2); cal=rng.uniform(size=100); seq=rng.uniform(size=20)
        loge,p=online_e(seq,cal,2)
        self.assertTrue(np.isfinite(loge).all()); self.assertTrue(((p>0)&(p<=1)).all())

    def test_particle_belief_normalized(self):
        lik=np.tile(np.array([[.8,.1,.1]]),(10,1)); b,ess=ParticleBelief(64,3).filter(lik)
        np.testing.assert_allclose(b.sum(1),1,atol=1e-6); self.assertTrue((ess>0).all())

    def test_partial_candidate_mechanisms(self):
        x=np.ones((20,6),np.float32); shifted=structured_graph_shift(x,[0,1],[4,5],[3],.2,1)
        self.assertEqual(shifted.shape,x.shape); self.assertFalse(np.array_equal(x,shifted))
        y=np.array([0,1]*10); env=np.repeat([0,1],10)
        model,q=fit_group_dro_binary(x,y,env,epochs=2)
        self.assertAlmostEqual(float(q.sum()),1.0,places=5)

    def test_enhancement_mixture_is_probability_distribution(self):
        a=np.array([[.8,.2],[.1,.9]]); b=np.array([[.3,.7],[.6,.4]])
        p=mix(a,b,.7)
        np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
        np.testing.assert_allclose(mix(a,b,1.0),a,atol=1e-12)

    def test_enhancement_selection_guard(self):
        path=ARTIFACTS/'protocol'/'enhancement_config.json'
        self.assertTrue(path.exists())
        cfg=json.loads(path.read_text(encoding='utf8'))
        self.assertFalse(cfg['test_labels_used_for_selection'])
        self.assertEqual(cfg['selection_labels'],'validation only')

    def test_reviewer_evidence_uses_frozen_models(self):
        path=ARTIFACTS/'protocol'/'reviewer_evidence_config.json'
        self.assertTrue(path.exists())
        cfg=json.loads(path.read_text(encoding='utf8'))
        self.assertFalse(cfg['test_labels_used_for_tuning'])
        self.assertEqual(cfg['splits'],'existing frozen manifests')
        self.assertEqual(cfg['models'],['AnytimeGraph-Guard','LightGBM','XGBoost'])


if __name__=='__main__': unittest.main()
