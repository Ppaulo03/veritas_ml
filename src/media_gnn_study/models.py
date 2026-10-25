
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
from .features import safe_unit_rows
from .core.graph import (
    build_knn_propagation,
    build_knn_edge_arrays,
    build_knn_edge_index,
    prune_tfidf_topk,
    prepare_hetero_term_incidence,
)


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, n_labels):
        super().__init__(); self.fc = nn.Linear(in_dim, n_labels)
    def forward(self, x, p=None): return self.fc(x)


class MLPGraphClassifier(nn.Module):
    def __init__(self, in_dim, n_labels, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden); self.fc2 = nn.Linear(hidden, n_labels); self.dropout = nn.Dropout(dropout)
    @staticmethod
    def _propagate(p, x):
        if p is None: return x
        return torch.sparse.mm(p, x) if p.is_sparse else p @ x
    def forward(self, x, p=None):
        h = self._propagate(p, x)
        h = self.dropout(F.relu(self.fc1(h)))
        h = self._propagate(p, h)
        return self.fc2(h)


def _ensure_torch_geometric():
    _ensure_package('torch_geometric', 'torch-geometric>=2.6,<3')
    import torch_geometric
    RUNTIME_INFO['torch_geometric'] = torch_geometric.__version__


class HomogeneousPyGClassifier(nn.Module):
    def __init__(self, in_dim, n_labels, operator, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        _ensure_torch_geometric()
        from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
        self.operator = str(operator)
        self.dropout = nn.Dropout(dropout)
        if self.operator == 'gcn':
            self.conv1 = GCNConv(in_dim, hidden, add_self_loops=True, normalize=True)
            self.conv2 = GCNConv(hidden, n_labels, add_self_loops=True, normalize=True)
        elif self.operator == 'graphsage':
            self.conv1 = SAGEConv(in_dim, hidden, aggr='mean', root_weight=True)
            self.conv2 = SAGEConv(hidden, n_labels, aggr='mean', root_weight=True)
        elif self.operator == 'gatv2':
            self.conv1 = GATv2Conv(in_dim, hidden, heads=1, concat=False, dropout=dropout, add_self_loops=True)
            self.conv2 = GATv2Conv(hidden, n_labels, heads=1, concat=False, dropout=dropout, add_self_loops=True)
        else:
            raise ValueError(operator)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.dropout(F.relu(h))
        return self.conv2(h, edge_index)


def fit_predict_homogeneous_operator(x_train, y_train, x_eval, model, seed, edge_train, edge_eval):
    torch.manual_seed(seed)
    xt=torch.tensor(x_train,dtype=torch.float32,device=DEVICE)
    yt=torch.tensor(y_train,dtype=torch.float32,device=DEVICE)
    xe=torch.tensor(np.concatenate([x_train,x_eval]),dtype=torch.float32,device=DEVICE)
    net=HomogeneousPyGClassifier(xt.shape[1],yt.shape[1],model).to(DEVICE)
    positives=yt.sum(0)
    if (positives==0).any() or (positives==len(yt)).any():
        raise ValueError('Classe degenerada no treino homogêneo.')
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=(len(yt)-positives)/positives)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    t0=time.perf_counter()
    for _ in range(EPOCHS):
        net.train(); opt.zero_grad(set_to_none=True)
        loss=loss_fn(net(xt,edge_train),yt)
        if not torch.isfinite(loss): raise RuntimeError(f'Loss {model} não finita.')
        loss.backward(); opt.step()
    train_s=time.perf_counter()-t0
    t1=time.perf_counter(); net.eval()
    with torch.inference_mode():
        prob=torch.sigmoid(net(xe,edge_eval)[len(xt):]).cpu().numpy()
    pred_s=time.perf_counter()-t1
    return prob,sum(p.numel() for p in net.parameters()),'parameters',train_s,pred_s


CLASSICAL_N_JOBS = CPU_THREADS


RF_CFG = dict(n_estimators=300, max_depth=None, min_samples_leaf=2, max_features='sqrt', class_weight='balanced_subsample', n_jobs=CLASSICAL_N_JOBS)


XGB_CFG = dict(
    n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=1.0, reg_lambda=1.0, objective='binary:logistic', eval_metric='logloss',
    tree_method='hist', n_jobs=CLASSICAL_N_JOBS, verbosity=0,
)


RUNTIME_INFO['sklearn'] = __import__('sklearn').__version__


def multilabel_metrics(y, prob, threshold=FIXED_THRESHOLD):
    pred = np.asarray(prob) >= threshold
    return {
        'f1_micro': f1_score(y, pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y, pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y, pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y, pred, average='micro', zero_division=0),
    }


def class_metrics(y, prob, classes, threshold=FIXED_THRESHOLD):
    pred=np.asarray(prob)>=threshold
    p,r,f,s=precision_recall_fscore_support(y,pred,average=None,zero_division=0)
    return pd.DataFrame({'class_name':list(classes),'precision':p,'recall':r,'f1':f,'support':s})


def fit_predict_neural(x_train, y_train, x_eval, model, seed, p_train=None, p_eval=None):
    torch.manual_seed(seed)
    xt = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    xe = torch.tensor(np.concatenate([x_train,x_eval]), dtype=torch.float32, device=DEVICE)
    if model == 'linear': net = LinearClassifier(xt.shape[1], yt.shape[1]).to(DEVICE); pt = pe = None
    elif model in ('mlp','gnn'):
        net = MLPGraphClassifier(xt.shape[1], yt.shape[1]).to(DEVICE)
        pt, pe = (p_train,p_eval) if model == 'gnn' else (None,None)
    else: raise ValueError(model)

    positives = yt.sum(0)
    if (positives == 0).any() or (positives == len(yt)).any(): raise ValueError('Classe degenerada no treino.')
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(len(yt)-positives)/positives)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        net.train(); opt.zero_grad(set_to_none=True)
        loss = loss_fn(net(xt, pt), yt)
        if not torch.isfinite(loss): raise RuntimeError('Loss não finita.')
        loss.backward(); opt.step()
    train_s = time.perf_counter()-t0
    t1 = time.perf_counter(); net.eval()
    with torch.inference_mode(): prob = torch.sigmoid(net(xe, pe)[len(xt):]).cpu().numpy()
    pred_s = time.perf_counter()-t1
    return prob, sum(p.numel() for p in net.parameters()), 'parameters', train_s, pred_s


def _positive_prob_from_rf(model, x):
    plist = model.predict_proba(x); cols = []
    for j,p in enumerate(plist):
        classes = np.asarray(model.classes_[j])
        cols.append(p[:,int(np.where(classes==1)[0][0])] if 1 in classes else np.zeros(len(x)))
    return np.column_stack(cols).astype(np.float32)


def fit_predict_classical(x_train, y_train, x_eval, model, seed):
    if model == 'rf':
        clf = RandomForestClassifier(random_state=seed, **RF_CFG)
        t0=time.perf_counter(); clf.fit(x_train,y_train); train_s=time.perf_counter()-t0
        t1=time.perf_counter(); prob=_positive_prob_from_rf(clf,x_eval); pred_s=time.perf_counter()-t1
        complexity=int(sum(t.tree_.node_count for t in clf.estimators_))
        return prob, complexity, 'tree_nodes', train_s, pred_s
    if model == 'xgb':
        _ensure_package('xgboost', 'xgboost>=2.0,<4')
        import xgboost as xgb
        estimators=[]; t0=time.perf_counter()
        for j in range(y_train.shape[1]):
            yj=y_train[:,j].astype(int); pos=max(1,int(yj.sum())); neg=max(1,len(yj)-pos)
            clf=xgb.XGBClassifier(random_state=seed+j, scale_pos_weight=neg/pos, **XGB_CFG)
            clf.fit(x_train,yj); estimators.append(clf)
        train_s=time.perf_counter()-t0
        t1=time.perf_counter(); prob=np.column_stack([clf.predict_proba(x_eval)[:,1] for clf in estimators]).astype(np.float32); pred_s=time.perf_counter()-t1
        complexity=int(sum(clf.get_booster().num_boosted_rounds() for clf in estimators))
        return prob, complexity, 'boosted_trees', train_s, pred_s
    raise ValueError(model)


class HeteroTermGNN(nn.Module):
    """GNN bipartida obra↔termo para classificação dos nós obra.

    Os termos são nós estruturais. Seus estados são agregados exclusivamente a
    partir das obras de treino; validação apenas consulta esses estados, evitando
    fluxo validação→termo→validação.
    """
    def __init__(self, work_dim, n_labels, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.work_encoder=nn.Linear(work_dim,hidden)
        self.work_to_term=nn.Linear(hidden,hidden,bias=False)
        self.self_path=nn.Linear(hidden,hidden,bias=False)
        self.term_to_work=nn.Linear(hidden,hidden,bias=False)
        self.dropout=nn.Dropout(dropout)
        self.out=nn.Linear(hidden,n_labels)

    @staticmethod
    def _term_state(h_train, col_normalized_train):
        return col_normalized_train.T @ h_train

    def _work_logits(self, h_work, row_normalized_incidence, term_state):
        msg=row_normalized_incidence @ self.term_to_work(term_state)
        h=F.relu(self.self_path(h_work)+msg)
        h=self.dropout(h)
        return self.out(h)

    def forward_train(self, x_train, row_train, col_train):
        h_train=F.relu(self.work_encoder(x_train))
        term_state=self._term_state(self.work_to_term(h_train),col_train)
        return self._work_logits(h_train,row_train,term_state)

    def forward_eval(self, x_train, x_eval, row_train, col_train, row_eval):
        h_train=F.relu(self.work_encoder(x_train))
        h_eval=F.relu(self.work_encoder(x_eval))
        term_state=self._term_state(self.work_to_term(h_train),col_train)
        return self._work_logits(h_eval,row_eval,term_state)


def fit_predict_hetero_term(x_train, y_train, x_eval, row_train, col_train, row_eval, seed):
    torch.manual_seed(seed)
    xt=torch.tensor(x_train,dtype=torch.float32,device=DEVICE)
    yt=torch.tensor(y_train,dtype=torch.float32,device=DEVICE)
    xe=torch.tensor(x_eval,dtype=torch.float32,device=DEVICE)
    rt=torch.tensor(row_train,dtype=torch.float32,device=DEVICE)
    ct=torch.tensor(col_train,dtype=torch.float32,device=DEVICE)
    re=torch.tensor(row_eval,dtype=torch.float32,device=DEVICE)
    net=HeteroTermGNN(xt.shape[1],yt.shape[1],hidden=HETERO_TERM_CFG['hidden'],dropout=HETERO_TERM_CFG['dropout']).to(DEVICE)
    positives=yt.sum(0)
    if (positives==0).any() or (positives==len(yt)).any():
        raise ValueError('Classe degenerada no treino heterogêneo.')
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=(len(yt)-positives)/positives)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    t0=time.perf_counter()
    for _ in range(EPOCHS):
        net.train(); opt.zero_grad(set_to_none=True)
        loss=loss_fn(net.forward_train(xt,rt,ct),yt)
        if not torch.isfinite(loss): raise RuntimeError('Loss heterogênea não finita.')
        loss.backward(); opt.step()
    train_s=time.perf_counter()-t0
    t1=time.perf_counter(); net.eval()
    with torch.inference_mode():
        prob=torch.sigmoid(net.forward_eval(xt,xe,rt,ct,re)).cpu().numpy()
    pred_s=time.perf_counter()-t1
    return prob,sum(p.numel() for p in net.parameters()),'parameters',train_s,pred_s


def _term_to_work_edges(row_incidence):
    arr=np.asarray(row_incidence,dtype=np.float32)
    work_idx,term_idx=np.nonzero(arr>0)
    if len(work_idx)==0:
        edge_index=torch.empty((2,0),dtype=torch.long,device=DEVICE)
        edge_attr=torch.empty((0,1),dtype=torch.float32,device=DEVICE)
    else:
        edge_index=torch.tensor(np.vstack([term_idx,work_idx]),dtype=torch.long,device=DEVICE)
        edge_attr=torch.tensor(arr[work_idx,term_idx,None],dtype=torch.float32,device=DEVICE)
    return edge_index,edge_attr


class HeteroTermGATv2(nn.Module):
    """Mantém termo-state train-only e troca apenas termo→obra por GATv2.

    Esta variante foi desenhada especificamente para testar a hipótese de sinal
    lexical difuso: a construção dos estados dos termos permanece idêntica ao
    baseline; somente a confiança em cada termo conectado à obra é aprendida.
    """
    def __init__(self, work_dim, n_labels, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        _ensure_torch_geometric()
        from torch_geometric.nn import GATv2Conv
        self.work_encoder=nn.Linear(work_dim,hidden)
        self.work_to_term=nn.Linear(hidden,hidden,bias=False)
        self.self_path=nn.Linear(hidden,hidden,bias=False)
        self.term_attention=GATv2Conv(
            (hidden,hidden),hidden,heads=1,concat=False,dropout=dropout,
            add_self_loops=False,edge_dim=1,
        )
        self.dropout=nn.Dropout(dropout)
        self.out=nn.Linear(hidden,n_labels)

    def _term_state(self,h_train,col_train):
        return col_train.T @ self.work_to_term(h_train)

    def _logits(self,h_work,term_state,edge_index,edge_attr):
        msg=self.term_attention((term_state,h_work),edge_index,edge_attr=edge_attr)
        h=self.dropout(F.relu(self.self_path(h_work)+msg))
        return self.out(h)

    def forward_train(self,x_train,col_train,edge_index,edge_attr):
        h_train=F.relu(self.work_encoder(x_train))
        term_state=self._term_state(h_train,col_train)
        return self._logits(h_train,term_state,edge_index,edge_attr)

    def forward_eval(self,x_train,x_eval,col_train,edge_index_eval,edge_attr_eval):
        h_train=F.relu(self.work_encoder(x_train))
        h_eval=F.relu(self.work_encoder(x_eval))
        term_state=self._term_state(h_train,col_train)
        return self._logits(h_eval,term_state,edge_index_eval,edge_attr_eval)


def fit_predict_hetero_term_gatv2(x_train,y_train,x_eval,row_train,col_train,row_eval,seed):
    torch.manual_seed(seed)
    xt=torch.tensor(x_train,dtype=torch.float32,device=DEVICE)
    yt=torch.tensor(y_train,dtype=torch.float32,device=DEVICE)
    xe=torch.tensor(x_eval,dtype=torch.float32,device=DEVICE)
    ct=torch.tensor(col_train,dtype=torch.float32,device=DEVICE)
    edge_tr,attr_tr=_term_to_work_edges(row_train)
    edge_ev,attr_ev=_term_to_work_edges(row_eval)
    net=HeteroTermGATv2(xt.shape[1],yt.shape[1],hidden=HETERO_TERM_CFG['hidden'],dropout=HETERO_TERM_CFG['dropout']).to(DEVICE)
    positives=yt.sum(0)
    if (positives==0).any() or (positives==len(yt)).any():
        raise ValueError('Classe degenerada no treino heterogêneo GATv2.')
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=(len(yt)-positives)/positives)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    t0=time.perf_counter()
    for _ in range(EPOCHS):
        net.train(); opt.zero_grad(set_to_none=True)
        loss=loss_fn(net.forward_train(xt,ct,edge_tr,attr_tr),yt)
        if not torch.isfinite(loss): raise RuntimeError('Loss heterogênea GATv2 não finita.')
        loss.backward(); opt.step()
    train_s=time.perf_counter()-t0
    t1=time.perf_counter(); net.eval()
    with torch.inference_mode():
        prob=torch.sigmoid(net.forward_eval(xt,xe,ct,edge_ev,attr_ev)).cpu().numpy()
    pred_s=time.perf_counter()-t1
    return prob,sum(p.numel() for p in net.parameters()),'parameters',train_s,pred_s
