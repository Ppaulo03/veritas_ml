
import os, sys, json, hashlib, re, time, warnings
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.ensemble import RandomForestClassifier
try:
    from IPython.display import display
except Exception:
    def display(x): print(x)

from .config import *
from .contracts import *
from .features import *
from .models import *

def combine_blocks(block_dict, sources):
    # Vetor zero é um estado válido para fontes esparsas: significa que a obra
    # não possui sinal conhecido naquele fold (ex.: ações removidas por min_df).
    # safe_unit_rows preserva esses vetores como zero; modelos não relacionais
    # podem consumi-los normalmente. A política especial de vizinhança fica no
    # construtor do grafo, que usa self-only para nós sem sinal.
    blocks=[safe_unit_rows(block_dict[s]) for s in sources]
    x=safe_unit_rows(np.concatenate(blocks,axis=1).astype(np.float32))
    if x.ndim!=2 or x.shape[1]==0 or not np.isfinite(x).all():
        raise ValueError(f'Bloco combinado inválido em {sources}')
    return x


def bundle_component_time(pack, bundle):
    names=[]
    for source in bundle: names.extend(SOURCE_COMPONENTS[source])
    unique=list(dict.fromkeys(names))
    fit=sum(float(pack['components'][c]['fit_seconds']) for c in unique)
    transform=sum(float(pack['components'][c]['transform_seconds']) for c in unique)
    return fit,transform,unique


def run_sanity_checks(dataset, feature_context, plan):
    for (tr,va),pack in zip(dataset.folds,feature_context.fold_packs):
        assert not set(dataset.frame.iloc[tr].grupo_franquia) & set(dataset.frame.iloc[va].grupo_franquia)
        for src in SOURCE_ORDER:
            assert pack['train'][src].shape[1]==pack['valid'][src].shape[1]

    tr,va=dataset.folds[0]
    xtr=combine_blocks(feature_context.fold_packs[0]['train'],('minilm',))
    xva=combine_blocks(feature_context.fold_packs[0]['valid'],('minilm',))
    p,pe,_=build_knn_propagation(xtr,xva,**plan.graph_config)
    assert p.shape==(len(tr),len(tr)) and pe.shape==(len(tr)+len(va),len(tr)+len(va))
    assert not pe[:len(tr),len(tr):].any()

    # Regressão: fontes esparsas podem ter linhas zero. Para modelos comuns isso
    # é ausência de sinal; para o grafo elas devem ficar self-only e nunca virar
    # vizinhas de outros nós.
    ztr=np.array([[1.,0.],[0.,0.],[0.,1.]],dtype=np.float32)
    zva=np.array([[0.,0.],[1.,1.]],dtype=np.float32)
    zp,zpe,zedges=build_knn_propagation(ztr,zva,k=2,self_weight=.9,min_similarity=None)
    assert np.count_nonzero(zp[1])==1 and zp[1,1]==1.0
    assert np.count_nonzero(zpe[len(ztr)])==1 and zpe[len(ztr),len(ztr)]==1.0
    assert 1 not in set(zedges.origem_treino_local.astype(int))

    torch.manual_seed(123)
    a=MLPGraphClassifier(8,4,hidden=5,dropout=0.0); b=MLPGraphClassifier(8,4,hidden=5,dropout=0.0)
    b.load_state_dict(a.state_dict()); z=torch.randn(7,8); ident=torch.eye(7).to_sparse()
    assert torch.allclose(a(z,None),b(z,ident),atol=1e-7)
    hetero_meta=sanity_check_hetero_term(dataset)
    assert hetero_meta['term_nodes']>0
    return True


def sanity_check_hetero_term(dataset):
    tr,va=dataset.folds[0]
    texts_tr=dataset.frame.iloc[tr].texto.tolist(); texts_va=dataset.frame.iloc[va].texto.tolist()
    _,a_tr,a_va,_,_=_fit_tfidf_base(texts_tr,texts_va,WORD_CFGS['word_1'])
    rt,ct,re,meta=prepare_hetero_term_incidence(a_tr,a_va)
    assert rt.shape==(len(tr),a_tr.shape[1])
    assert ct.shape==rt.shape and re.shape==(len(va),a_tr.shape[1])
    # colunas com incidência têm soma 1 na direção obra→termo agregadora.
    nz=a_tr.toarray().sum(axis=0)>1e-12
    if np.any(nz): assert np.allclose(ct[:,nz].sum(axis=0),1.0,atol=1e-5)
    assert meta['term_nodes']==a_tr.shape[1]
    return meta
