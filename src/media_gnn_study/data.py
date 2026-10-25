
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

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_text(text):
    text = re.sub(r'\[\s*(?:Escrito por MAL Reescrever|Written by MAL Rewrite)\s*\]', ' ', str(text), flags=re.I)
    return re.sub(r'\s+', ' ', text).strip()


def load_development(data_path=DATA, split_path=PARTS):
    if sha256(data_path) != DATA_SHA:
        raise ValueError('Dataset diferente da versão congelada.')
    if sha256(split_path) != SPLIT_SHA:
        raise ValueError('Partições diferentes da versão congelada.')

    raw = pd.read_csv(data_path, sep=';', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    parts = pd.read_csv(split_path, dtype=str)
    assert raw.Id.is_unique and parts.Id.is_unique and set(raw.Id) == set(parts.Id)
    assert set(parts.particao) == {'desenvolvimento', 'auditoria'}

    g_dev = set(parts.loc[parts.particao == 'desenvolvimento', 'grupo_franquia'])
    g_aud = set(parts.loc[parts.particao == 'auditoria', 'grupo_franquia'])
    assert not (g_dev & g_aud)

    dev_ids = set(parts.loc[parts.particao == 'desenvolvimento', 'Id'])
    dev = raw.loc[raw.Id.isin(dev_ids), ['Id','Nome','Sinopse','Origem',TARGET]].copy()
    dev = dev.merge(parts[['Id','grupo_franquia']], on='Id', validate='one_to_one')
    dev = dev.sort_values('Id').reset_index(drop=True)
    dev['texto'] = dev.Sinopse.map(clean_text)
    assert len(dev) == 304 and dev.texto.str.len().gt(0).all()

    labels = [[x.strip() for x in s.split(',') if x.strip()] for s in dev[TARGET]]
    unknown = sorted(set(x for row in labels for x in row) - set(CLASSES))
    assert not unknown, f'Classes não previstas: {unknown}'
    mlb = MultiLabelBinarizer(classes=CLASSES)
    y = mlb.fit_transform(labels).astype(np.float32)
    return dev, y, parts


def make_grouped_folds(groups, n_splits=CV_SPLITS, seed=CV_SEED):
    groups = pd.Series(groups, dtype=str).reset_index(drop=True)
    sizes = groups.value_counts().rename_axis('group').reset_index(name='size')
    rng = np.random.default_rng(seed)
    sizes['tie'] = rng.random(len(sizes))
    sizes = sizes.sort_values(['size','tie'], ascending=[False,True]).reset_index(drop=True)

    fold_sizes = np.zeros(n_splits, dtype=int)
    assignment = {}
    for row in sizes.itertuples(index=False):
        f = int(np.argmin(fold_sizes))
        assignment[row.group] = f
        fold_sizes[f] += int(row.size)

    fold_id = groups.map(assignment).to_numpy()
    folds = []
    for f in range(n_splits):
        va = np.where(fold_id == f)[0]
        tr = np.where(fold_id != f)[0]
        assert not set(groups.iloc[tr]) & set(groups.iloc[va])
        folds.append((tr, va))
    return folds, fold_id


def build_dataset_context(data_path=None, split_path=None, persist_folds=True):
    ensure_runtime_dirs()
    data_path = Path(data_path or DATA)
    split_path = Path(split_path or PARTS)
    dev, y, partitions = load_development(data_path, split_path)
    folds, fold_id = make_grouped_folds(dev.grupo_franquia)
    dataset = DatasetContext(
        frame=dev, targets=y, classes=tuple(CLASSES), folds=tuple(folds), fold_id=fold_id,
        partitions=partitions,
        metadata={
            'data_sha256': DATA_SHA, 'split_sha256': SPLIT_SHA,
            'n_development': int(len(dev)),
            'n_audit_reserved': int((partitions.particao == 'auditoria').sum()),
            'target_column': TARGET, 'group_column': 'grupo_franquia',
        },
    )
    if persist_folds:
        pd.DataFrame({'Id':dataset.frame.Id,'fold':dataset.fold_id+1}).to_csv(STUDY_DIR/'folds.csv',index=False)
    return dataset


def fold_summary(dataset):
    rows=[]
    for fold,(tr,va) in enumerate(dataset.folds,1):
        support=dataset.targets[tr].sum(0)
        rows.append({'fold':fold,'n_treino':len(tr),'n_validacao':len(va),
                     'grupos_validacao':dataset.frame.iloc[va].grupo_franquia.nunique(),
                     'menor_suporte_treino':int(support.min())})
    return pd.DataFrame(rows)
