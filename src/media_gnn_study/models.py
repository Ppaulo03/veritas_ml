
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

def build_knn_propagation(x_train, x_query, k=5, self_weight=.90, min_similarity=None):
    xt, xq = safe_unit_rows(x_train), safe_unit_rows(x_query)
    n, m = len(xt), len(xq)
    if n < 2 or k < 1 or not (0 <= self_weight <= 1):
        raise ValueError('Configuração de grafo inválida.')

    # Fontes esparsas podem produzir vetor zero quando não há sinal conhecido
    # no fold. Esses nós não devem ganhar vizinhos arbitrários por empate em 0.
    # Política: vetor zero => self-only; vetor zero de treino nunca é candidato.
    eps = 1e-10
    train_has_signal = np.linalg.norm(xt, axis=1) >= eps
    query_has_signal = np.linalg.norm(xq, axis=1) >= eps
    sim_train = xt @ xt.T
    sim_query = xq @ xt.T
    p = np.eye(n, dtype=np.float32)
    pe = np.zeros((n+m, n+m), dtype=np.float32)
    edges = []

    def selected(row, own=None):
        candidates = np.flatnonzero(train_has_signal)
        if own is not None:
            candidates = candidates[candidates != own]
        if len(candidates) == 0:
            return candidates
        order = candidates[np.argsort(-row[candidates], kind='stable')]
        order = order[:min(k, len(order))]
        if min_similarity is not None:
            order = order[row[order] >= min_similarity]
        return order

    for i in range(n):
        nbr = selected(sim_train[i], own=i) if train_has_signal[i] else np.empty(0, dtype=int)
        p[i,:] = 0
        if len(nbr) == 0:
            p[i,i] = 1.0
        else:
            p[i,i] = self_weight
            p[i,nbr] = (1-self_weight)/len(nbr)
        for rank,j in enumerate(nbr,1):
            edges.append(('treino',i,int(j),rank,float(sim_train[i,j])))

    pe[:n,:n] = p
    for q in range(m):
        nbr = selected(sim_query[q]) if query_has_signal[q] else np.empty(0, dtype=int)
        row = n+q
        if len(nbr) == 0:
            pe[row,row] = 1.0
        else:
            pe[row,row] = self_weight
            pe[row,nbr] = (1-self_weight)/len(nbr)
        for rank,j in enumerate(nbr,1):
            edges.append(('validacao',q,int(j),rank,float(sim_query[q,j])))

    assert np.allclose(p.sum(1),1) and np.allclose(pe.sum(1),1)
    assert not pe[:n,n:].any()
    # Zero-signal nodes are guaranteed to be self-only and are never neighbors.
    zero_train = np.flatnonzero(~train_has_signal)
    if len(zero_train):
        assert np.allclose(p[zero_train, zero_train], 1.0)
        assert not any(int(j) in set(zero_train.tolist()) for _,_,j,_,_ in edges)
    zero_query = np.flatnonzero(~query_has_signal)
    for q in zero_query:
        row=n+int(q)
        assert pe[row,row] == 1.0 and np.count_nonzero(pe[row]) == 1

    return p, pe, pd.DataFrame(edges, columns=['subset','destino_local','origem_treino_local','rank','cosseno'])


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


def build_knn_edge_index(x_train, x_query, k=5, min_similarity=None):
    """Mesma seleção kNN da GNN fixa, mas retorna apenas arestas j→i.

    Em validação, nós de validação recebem mensagens somente de treino.
    Self-loops são responsabilidade de cada operador (GCN/GATv2); GraphSAGE
    usa seu root path nativo. Isso preserva a topologia de vizinhança sem
    importar o self_weight=0.9 específico do baseline fixo.
    """
    _, _, edges = build_knn_propagation(
        x_train, x_query, k=k, self_weight=0.0, min_similarity=min_similarity
    )
    n = len(x_train)
    src, dst = [], []
    for row in edges.itertuples(index=False):
        src.append(int(row.origem_treino_local))
        dst.append(int(row.destino_local) if row.subset == 'treino' else n + int(row.destino_local))
    if not src:
        return torch.empty((2,0), dtype=torch.long, device=DEVICE)
    return torch.tensor([src,dst], dtype=torch.long, device=DEVICE)


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


def prune_tfidf_topk(a, top_k=None):
    """Mantém os top-k pesos TF-IDF de cada obra; não usa rótulos."""
    if top_k is None:
        return a.copy()
    x=a.tocsr(copy=True)
    k=int(top_k)
    if k < 1:
        raise ValueError('top_k deve ser >= 1 ou None.')
    for i in range(x.shape[0]):
        lo,hi=x.indptr[i],x.indptr[i+1]
        if hi-lo <= k:
            continue
        vals=x.data[lo:hi]
        keep=np.argpartition(vals,-k)[-k:]
        mask=np.ones(hi-lo,dtype=bool); mask[keep]=False
        vals[mask]=0.0
    x.eliminate_zeros()
    return x


def prepare_hetero_term_incidence(a_train, a_eval, top_k=None):
    """Incidências TF-IDF normalizadas, opcionalmente podadas por obra."""
    a_train=prune_tfidf_topk(a_train,top_k)
    a_eval=prune_tfidf_topk(a_eval,top_k)
    train=np.asarray(a_train.toarray(),dtype=np.float32)
    eval_=np.asarray(a_eval.toarray(),dtype=np.float32)
    # obra→termo para construir estado dos termos: apenas obras de treino.
    col_sum=train.sum(axis=0,keepdims=True)
    col_norm=np.divide(train,col_sum,out=np.zeros_like(train),where=col_sum>1e-12)
    # termo→obra: pesos TF-IDF normalizados dentro de cada obra.
    row_sum_tr=train.sum(axis=1,keepdims=True)
    row_sum_ev=eval_.sum(axis=1,keepdims=True)
    row_train=np.divide(train,row_sum_tr,out=np.zeros_like(train),where=row_sum_tr>1e-12)
    row_eval=np.divide(eval_,row_sum_ev,out=np.zeros_like(eval_),where=row_sum_ev>1e-12)
    meta={
        'term_nodes':int(train.shape[1]),
        'incidence_train':int(np.count_nonzero(train)),
        'incidence_eval':int(np.count_nonzero(eval_)),
        'zero_term_train_rows':int(np.sum(row_sum_tr[:,0]<=1e-12)),
        'zero_term_eval_rows':int(np.sum(row_sum_ev[:,0]<=1e-12)),
        'term_top_k':None if top_k is None else int(top_k),
    }
    return row_train,col_norm,row_eval,meta


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
