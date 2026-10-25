
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
from .features import SOURCE_ORDER

def _dedupe_registry(rows):
    seen={}; out=[]
    for family,bundle in rows:
        bundle=tuple(bundle); key='+'.join(bundle)
        if key not in seen:
            seen[key]=family; out.append({'family':family,'bundle':bundle,'features':key})
    return out

CORE_SOURCES = ('minilm','word_1_2','entities','emotion','topics')
registry_rows=[]
for r in range(1,len(CORE_SOURCES)+1):
    for combo in combinations(CORE_SOURCES,r): registry_rows.append(('core_factorial',combo))

registry_rows += [
    ('lexical',('word_1',)), ('lexical',('word_1_3',)), ('lexical',('char_3_5',)),
    ('lexical',('word_1_2','char_3_5')), ('lexical',('word_1_3','char_3_5')),
    ('lexical',('minilm','word_1')), ('lexical',('minilm','word_1_3')), ('lexical',('minilm','char_3_5')),
    ('lexical',('minilm','word_1_2','char_3_5')), ('lexical',('minilm','word_1_3','char_3_5')),
]
registry_rows += [
    ('narrative',('actions',)), ('narrative',('linguistic',)),
    ('narrative',('actions','emotion')), ('narrative',('actions','topics')),
    ('narrative',('actions','emotion','topics')), ('narrative',('actions','linguistic')),
    ('narrative',('linguistic','emotion')), ('narrative',('linguistic','topics')),
    ('narrative',('minilm','actions')), ('narrative',('minilm','linguistic')),
    ('narrative',('minilm','actions','emotion')), ('narrative',('minilm','actions','topics')),
    ('narrative',('minilm','actions','linguistic')), ('narrative',('minilm','actions','emotion','topics')),
]

BUNDLE_REGISTRY = _dedupe_registry(registry_rows)
for row in BUNDLE_REGISTRY:
    assert all(s in SOURCE_ORDER for s in row['bundle'])
BUNDLE_TABLE=pd.DataFrame([{**r,'bundle':'|'.join(r['bundle'])} for r in BUNDLE_REGISTRY])

SPECIAL_EXPERIMENTS = (
    # 1) Homogênea: mesmo vetor, mesmo kNN, muda somente o operador.
    {
        'id':'homogeneous_gcn_minilm_word1_v1','adapter':'homogeneous_operator','family':'homogeneous_operator',
        'features':'minilm+word_1','bundle':('minilm','word_1'),'model_id':'gcn','fontes':'minilm|word_1','n_fontes':2,
        'enabled':True,'description':'GCNConv no mesmo kNN indutivo usado pelo baseline Fixed-kNN GNN',
    },
    {
        'id':'homogeneous_graphsage_minilm_word1_v1','adapter':'homogeneous_operator','family':'homogeneous_operator',
        'features':'minilm+word_1','bundle':('minilm','word_1'),'model_id':'graphsage','fontes':'minilm|word_1','n_fontes':2,
        'enabled':True,'description':'SAGEConv(mean) no mesmo kNN indutivo usado pelo baseline Fixed-kNN GNN',
    },
    {
        'id':'homogeneous_gatv2_minilm_word1_v1','adapter':'homogeneous_operator','family':'homogeneous_operator',
        'features':'minilm+word_1','bundle':('minilm','word_1'),'model_id':'gatv2','fontes':'minilm|word_1','n_fontes':2,
        'enabled':True,'description':'GATv2Conv no mesmo kNN indutivo; atenção aprende quanto confiar em cada vizinho',
    },

    # 2) Heterogênea: baseline atual e testes da hipótese de sinal lexical difuso.
    {
        'id':'hetero_term_gnn_v1','adapter':'hetero_term','family':'heterogeneous',
        'features':HETERO_TERM_CFG['representation_name'],'model_id':'hetero_term_gnn','fontes':'minilm|word_1_terms','n_fontes':2,
        'top_k_terms':None,'enabled':True,
        'description':'Baseline atual: MiniLM nas obras + todos os nós termo, incidência TF-IDF train-only',
    },
    *tuple({
        'id':f'hetero_term_gnn_top{k}_v1','adapter':'hetero_term','family':'heterogeneous_diffusion',
        'features':f"{HETERO_TERM_CFG['representation_name']}_top{k}",'model_id':'hetero_term_gnn',
        'fontes':'minilm|word_1_terms','n_fontes':2,'top_k_terms':k,'enabled':True,
        'description':f'Mesma hetero, mantendo apenas os top-{k} termos TF-IDF por obra',
    } for k in HETERO_DIFFUSION_CFG['top_k_values']),
    {
        'id':'hetero_term_gatv2_v1','adapter':'hetero_term','family':'heterogeneous_attention',
        'features':HETERO_TERM_CFG['representation_name'],'model_id':'hetero_term_gatv2','fontes':'minilm|word_1_terms','n_fontes':2,
        'top_k_terms':None,'enabled':True,
        'description':'Estados dos termos train-only; GATv2 aprende a agregação termo→obra usando TF-IDF como atributo de aresta',
    },
)




def build_experiment_plan(dataset=None, persist_registry=True):
    plan=ExperimentPlan(
        registry=tuple(BUNDLE_REGISTRY), model_ids=tuple(MODELS),
        model_seeds={k:tuple(v) for k,v in MODEL_SEEDS.items()},
        model_metadata=MODEL_METADATA, primary_metrics=PRIMARY_METRICS,
        reported_metrics=REPORTED_METRICS, class_metrics=CLASS_METRICS,
        metric_labels=METRIC_LABELS, reference_policy=REFERENCE_POLICY,
        graph_config=GRAPH_CFG, diagnostics=RELATIONAL_DIAGNOSTICS,
        special_experiments=tuple(x for x in SPECIAL_EXPERIMENTS if x.get('enabled',True)),
        registered_models=tuple(MODEL_METADATA),
    )
    if persist_registry:
        ensure_runtime_dirs(); BUNDLE_TABLE.to_csv(STUDY_DIR/'bundle_registry.csv',index=False)
    return plan


def plan_summary(plan, dataset=None):
    main_conditions=len(plan.registry)*len(plan.model_ids)
    special_conditions=len(plan.special_experiments)
    d={'bundles':len(plan.registry),'main_models':len(plan.model_ids),'main_conditions':main_conditions,
       'special_conditions':special_conditions,'conditions':main_conditions+special_conditions}
    if dataset is not None:
        main_fits=sum(len(dataset.folds)*len(plan.model_seeds[m]) for m in plan.model_ids)*len(plan.registry)
        special_fits=sum(len(dataset.folds)*len(plan.model_seeds[e['model_id']]) for e in plan.special_experiments)
        d['fits']=main_fits+special_fits
    return d
