
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
from .sanity import combine_blocks
from .engine import *

def _metric_condition_summary(experiment_result):
    obs=experiment_result.metric_observations
    fold=(obs.groupby(['condition_id','metric','fold'],as_index=False).value.mean())
    return fold.groupby(['condition_id','metric'],as_index=False).agg(mean=('value','mean'),sd=('value','std'))


def run_relational_diagnostics(dataset, feature_context, experiment_result, plan):
    if not plan.diagnostics:
        return {}
    conditions=experiment_result.conditions
    summary=_metric_condition_summary(experiment_result)
    registry={r['features']:tuple(r['bundle']) for r in plan.registry}
    outputs={}

    for spec in plan.diagnostics:
        rel_model=spec['relational_model']; base_model=spec['baseline_model']; metric=spec['metric']
        available=set(conditions.model_id)
        if rel_model not in available or base_model not in available:
            continue

        edge_rows=[]
        for representation,bundle in registry.items():
            for fold,((tr,va),pack) in enumerate(zip(dataset.folds,feature_context.fold_packs),1):
                xtr=combine_blocks(pack['train'],bundle); xva=combine_blocks(pack['valid'],bundle)
                _,_,edges=build_knn_propagation(xtr,xva,**plan.graph_config)
                e=edges[edges.subset=='validacao']
                for row in e.itertuples(index=False):
                    yq=dataset.targets[va[int(row.destino_local)]].astype(bool)
                    yn=dataset.targets[tr[int(row.origem_treino_local)]].astype(bool)
                    inter=np.logical_and(yq,yn).sum(); union=np.logical_or(yq,yn).sum()
                    edge_rows.append({
                        'representation_name':representation,'fold':fold,'rank':int(row.rank),
                        'cosine':float(row.cosseno),'any_label_overlap':float(inter>0),
                        'jaccard':float(inter/union if union else 1.0),'shared_labels':int(inter),
                    })
        edges=pd.DataFrame(edge_rows)
        hom=(edges.groupby('representation_name',as_index=False).agg(
            mean_cosine=('cosine','mean'), any_label_overlap=('any_label_overlap','mean'),
            mean_jaccard=('jaccard','mean'), mean_shared_labels=('shared_labels','mean'),
        ))

        score=summary[summary.metric==metric].merge(conditions[['condition_id','representation_name','model_id']],on='condition_id',validate='many_to_one')
        rel=score[score.model_id==rel_model][['representation_name','mean']].rename(columns={'mean':'relational_score'})
        base=score[score.model_id==base_model][['representation_name','mean']].rename(columns={'mean':'baseline_score'})
        view=hom.merge(rel,on='representation_name').merge(base,on='representation_name')
        view['delta']=view.relational_score-view.baseline_score
        view['diagnostic_id']=spec['id']
        view['comparison_label']=f"{plan.model_metadata[rel_model]['display_name']} − {plan.model_metadata[base_model]['display_name']}"
        view['metric']=metric
        outputs[spec['id']]=view
        outputs[f"{spec['id']}__edges"]=edges
    return outputs


LEARNING_CURVE_CONFIG = {
    'id':'learning_curve',
    'version':'learning_curve_nested_group_balanced_v1',
    'representation_name':'minilm+word_1',
    'bundle':('minilm','word_1'),
    'models':('mlp','gnn'),
    'fractions':(0.20,0.40,0.60,0.80,1.00),
    'subset_seed':CV_SEED + 1000,
    'subset_policy':'nested_group_greedy_multilabel_balance_train_only',
    'validation_policy':'same frozen outer validation fold for every fraction',
    'feature_policy':'MiniLM frozen; word_1 TF-IDF/SVD refit on each training subset',
}


def _learning_curve_group_order(dataset, tr, seed):
    tr=np.asarray(tr,dtype=int)
    group_values=dataset.frame.iloc[tr].grupo_franquia.astype(str).to_numpy()
    full_counts=dataset.targets[tr].sum(0).astype(float)
    full_prev=full_counts/max(len(tr),1)
    weights=1/np.sqrt(np.maximum(full_prev,1/max(len(tr),1)))
    weights=weights/max(weights.mean(),1e-12)

    groups=[]
    for group in sorted(np.unique(group_values)):
        idx=tr[group_values==group]
        groups.append({
            'group':str(group), 'indices':idx,
            'counts':dataset.targets[idx].sum(0).astype(float), 'n':int(len(idx)),
        })
    rng=np.random.default_rng(seed)
    tie={g['group']:float(rng.random()) for g in groups}
    remaining=list(groups); order=[]
    selected_counts=np.zeros(dataset.targets.shape[1],dtype=float); selected_n=0

    while remaining:
        best_idx=None; best_score=None
        for i,item in enumerate(remaining):
            new_counts=selected_counts+item['counts']; new_n=selected_n+item['n']
            prevalence=new_counts/max(new_n,1)
            distribution_error=float(np.mean(weights*np.abs(prevalence-full_prev)))
            missing_penalty=float(np.mean((new_counts==0)&(full_counts>0)))
            score=distribution_error + 0.35*missing_penalty + 1e-9*tie[item['group']]
            if best_score is None or score<best_score:
                best_idx=i; best_score=score
        item=remaining.pop(best_idx)
        order.append(item); selected_counts+=item['counts']; selected_n+=item['n']
    return order


def _learning_curve_nested_subsets(dataset, tr, fractions, seed):
    order=_learning_curve_group_order(dataset,tr,seed)
    selected=[]; cursor=0; outputs=[]
    n_outer=len(tr)
    for fraction in fractions:
        target=n_outer if float(fraction)>=1 else int(np.ceil(float(fraction)*n_outer))
        while cursor<len(order):
            support=dataset.targets[np.asarray(selected,dtype=int)].sum(0) if selected else np.zeros(dataset.targets.shape[1])
            if len(selected)>=target and np.all(support>0):
                break
            selected.extend(order[cursor]['indices'].tolist()); cursor+=1
        subset=np.asarray(sorted(selected),dtype=int)
        support=dataset.targets[subset].sum(0)
        if np.any(support==0):
            raise RuntimeError(f'Learning curve sem suporte para alguma classe em {fraction:.0%}.')
        outputs.append({
            'target_fraction':float(fraction), 'fraction_percent':int(round(100*float(fraction))),
            'actual_fraction':float(len(subset)/n_outer), 'indices':subset,
            'n_train':int(len(subset)),
            'n_groups':int(dataset.frame.iloc[subset].grupo_franquia.nunique()),
            'support':support.astype(int),
        })
    for a,b in zip(outputs,outputs[1:]):
        assert set(a['indices']).issubset(set(b['indices']))
    return outputs


def _learning_curve_features(dataset, feature_context, fold_index, tr, va, subset_indices):
    pack=feature_context.fold_packs[fold_index]
    outer_position={int(idx):i for i,idx in enumerate(np.asarray(tr,dtype=int))}
    local=np.asarray([outer_position[int(idx)] for idx in subset_indices],dtype=int)
    minilm_tr=pack['train']['minilm'][local]
    minilm_va=pack['valid']['minilm']

    texts_tr=dataset.frame.iloc[subset_indices].texto.tolist()
    texts_va=dataset.frame.iloc[va].texto.tolist()
    _,a_tr,a_va,fit_base,transform_base=_fit_tfidf_base(texts_tr,texts_va,WORD_CFGS['word_1'])
    word_tr,word_va,_,fit_svd,transform_svd=_fit_svd(a_tr,a_va,LEX_SVD_DIM,seed=CV_SEED)

    t0=time.perf_counter()
    xtr=combine_blocks({'minilm':minilm_tr,'word_1':word_tr},('minilm','word_1'))
    xva=combine_blocks({'minilm':minilm_va,'word_1':word_va},('minilm','word_1'))
    fusion_s=time.perf_counter()-t0
    return xtr,xva,{
        'feature_fit_seconds':float(fit_base+fit_svd),
        'feature_transform_seconds':float(transform_base+transform_svd),
        'feature_fusion_seconds':float(fusion_s),
        'word_1_vocab':int(a_tr.shape[1]), 'word_1_dim':int(word_tr.shape[1]),
    }


def run_learning_curve_diagnostic(dataset, feature_context, experiment_result, plan, spec=LEARNING_CURVE_CONFIG):
    models=tuple(m for m in spec['models'] if m in plan.model_ids)
    if not models:
        return {}
    for model in models:
        if model not in ('mlp','gnn'):
            raise ValueError('Learning curve atual usa a interface neural MLP/GNN.')

    metrics_path=STUDY_DIR/'learning_curve_metrics_partial.csv'
    class_path=STUDY_DIR/'learning_curve_class_metrics_partial.csv'
    if metrics_path.is_file():
        raw=pd.read_csv(metrics_path)
        if 'diag_version' not in raw or set(raw.diag_version)!={spec['version']}:
            raw=pd.DataFrame()
    else:
        raw=pd.DataFrame()
    if class_path.is_file():
        raw_class=pd.read_csv(class_path)
        if 'diag_version' not in raw_class or set(raw_class.diag_version)!={spec['version']}:
            raw_class=pd.DataFrame()
    else:
        raw_class=pd.DataFrame()

    done=set()
    if not raw.empty:
        done=set(zip(raw.fold.astype(int),raw.fraction_percent.astype(int),raw.model_id.astype(str),raw.seed.astype(int)))

    subset_rows=[]; support_rows=[]
    total_units=len(dataset.folds)*len(spec['fractions']); unit=0
    for fold_index,(tr,va) in enumerate(dataset.folds):
        subsets=_learning_curve_nested_subsets(dataset,tr,spec['fractions'],spec['subset_seed']+fold_index)
        for subset in subsets:
            unit+=1; pct=subset['fraction_percent']
            for class_name,support in zip(dataset.classes,subset['support']):
                support_rows.append({
                    'fold':fold_index+1,'target_fraction':subset['target_fraction'],'fraction_percent':pct,
                    'actual_fraction':subset['actual_fraction'],'n_train':subset['n_train'],
                    'class_name':class_name,'train_support':int(support),
                })
            subset_rows.append({
                'fold':fold_index+1,'target_fraction':subset['target_fraction'],'fraction_percent':pct,
                'actual_fraction':subset['actual_fraction'],'n_train':subset['n_train'],'n_groups':subset['n_groups'],
            })

            expected_keys=[(fold_index+1,pct,m,int(seed)) for m in models for seed in plan.model_seeds[m]]
            if all(k in done for k in expected_keys):
                print(f"[LC {unit:02d}/{total_units}] fold {fold_index+1} | {pct:3d}% | checkpoint")
                continue

            xtr,xva,feature_meta=_learning_curve_features(dataset,feature_context,fold_index,tr,va,subset['indices'])
            pt=pe=None; graph_s=0.0
            if any(plan.model_metadata.get(m,{}).get('role')=='relational' for m in models):
                tg=time.perf_counter(); p_train,p_eval,_=build_knn_propagation(xtr,xva,**plan.graph_config); graph_s=time.perf_counter()-tg
                pt=torch.tensor(p_train,dtype=torch.float32,device=DEVICE).to_sparse().coalesce()
                pe=torch.tensor(p_eval,dtype=torch.float32,device=DEVICE).to_sparse().coalesce()

            new_rows=[]; new_class=[]
            for model in models:
                use_graph=plan.model_metadata.get(model,{}).get('role')=='relational'
                for seed in plan.model_seeds[model]:
                    key=(fold_index+1,pct,model,int(seed))
                    if key in done:
                        continue
                    prob,complexity,kind,train_s,predict_s=fit_predict_neural(
                        xtr,dataset.targets[subset['indices']],xva,model,int(seed),pt if use_graph else None,pe if use_graph else None
                    )
                    score=multilabel_metrics(dataset.targets[va],prob)
                    row={
                        'diag_version':spec['version'],'representation_name':spec['representation_name'],
                        'fold':fold_index+1,'target_fraction':subset['target_fraction'],'fraction_percent':pct,
                        'actual_fraction':subset['actual_fraction'],'n_train':subset['n_train'],'n_groups':subset['n_groups'],
                        'model_id':model,'seed':int(seed),'input_dim':int(xtr.shape[1]),
                        'word_1_vocab':feature_meta['word_1_vocab'],'word_1_dim':feature_meta['word_1_dim'],
                        'feature_fit_seconds':feature_meta['feature_fit_seconds'],
                        'feature_transform_seconds':feature_meta['feature_transform_seconds'],
                        'feature_fusion_seconds':feature_meta['feature_fusion_seconds'],
                        'graph_build_seconds':graph_s if use_graph else 0.0,
                        'train_seconds':float(train_s),'predict_seconds':float(predict_s),
                        **score,
                    }
                    new_rows.append(row)
                    cm=class_metrics(dataset.targets[va],prob,dataset.classes)
                    cm['diag_version']=spec['version']; cm['representation_name']=spec['representation_name']
                    cm['fold']=fold_index+1; cm['target_fraction']=subset['target_fraction']; cm['fraction_percent']=pct
                    cm['actual_fraction']=subset['actual_fraction']; cm['n_train']=subset['n_train']; cm['model_id']=model; cm['seed']=int(seed)
                    new_class.append(cm)
                    done.add(key)

            if new_rows:
                raw=pd.concat([raw,pd.DataFrame(new_rows)],ignore_index=True)
                raw_class=pd.concat([raw_class,*new_class],ignore_index=True)
                raw.to_csv(metrics_path,index=False); raw_class.to_csv(class_path,index=False)
            print(f"[LC {unit:02d}/{total_units}] fold {fold_index+1} | {pct:3d}% | n={subset['n_train']} | pronto")

    expected=len(dataset.folds)*len(spec['fractions'])*sum(len(plan.model_seeds[m]) for m in models)
    if len(raw)!=expected:
        raise AssertionError(f'Learning curve incompleta: {len(raw)} linhas; esperado {expected}.')

    # O ponto de 100% deve reproduzir a mesma condição da matriz principal.
    # Isso valida que a learning curve não mudou silenciosamente o pipeline.
    if experiment_result is not None and not experiment_result.raw_metrics.empty:
        base=experiment_result.raw_metrics[
            (experiment_result.raw_metrics.features.astype(str)==str(spec['representation_name'])) &
            (experiment_result.raw_metrics.modelo.astype(str).isin(models))
        ].copy()
        lc100=raw[raw.fraction_percent==100].copy()
        if not base.empty and not lc100.empty:
            merged=lc100.merge(
                base[['modelo','fold','seed',*plan.primary_metrics]],
                left_on=['model_id','fold','seed'],right_on=['modelo','fold','seed'],
                suffixes=('_lc','_main'),validate='one_to_one'
            )
            for metric in plan.primary_metrics:
                diff=np.abs(merged[f'{metric}_lc']-merged[f'{metric}_main'])
                if len(diff) and float(diff.max())>1e-7:
                    raise AssertionError(f'Learning curve 100% divergiu da matriz principal em {metric}: {float(diff.max())}')
            print('Learning curve 100% reproduz a matriz principal: OK')

    raw.to_csv(STUDY_DIR/'learning_curve_metrics.csv',index=False)
    raw_class.to_csv(STUDY_DIR/'learning_curve_class_metrics.csv',index=False)
    subsets_df=pd.DataFrame(subset_rows).drop_duplicates().sort_values(['fold','fraction_percent'])
    support_df=pd.DataFrame(support_rows).drop_duplicates().sort_values(['fold','fraction_percent','class_name'])
    subsets_df.to_csv(STUDY_DIR/'learning_curve_subsets.csv',index=False)
    support_df.to_csv(STUDY_DIR/'learning_curve_subset_support.csv',index=False)

    metric_long=raw.melt(
        id_vars=['fold','fraction_percent','target_fraction','actual_fraction','n_train','model_id'],
        value_vars=[m for m in plan.primary_metrics if m in raw.columns],var_name='metric',value_name='value'
    )
    fold_metric=(metric_long.groupby(['model_id','fraction_percent','metric','fold'],as_index=False).agg(
        value=('value','mean'),n_train=('n_train','first'),actual_fraction=('actual_fraction','first')
    ))
    global_summary=(fold_metric.groupby(['model_id','fraction_percent','metric'],as_index=False).agg(
        mean=('value','mean'),sd=('value','std'),n_train_mean=('n_train','mean'),
        n_train_min=('n_train','min'),n_train_max=('n_train','max'),actual_fraction_mean=('actual_fraction','mean')
    ))
    global_summary['model_name']=global_summary.model_id.map(lambda m: plan.model_metadata.get(m,{}).get('display_name',m))

    class_long=raw_class.melt(
        id_vars=['fold','fraction_percent','target_fraction','actual_fraction','n_train','model_id','class_name'],
        value_vars=[m for m in ('precision','recall','f1') if m in raw_class.columns],var_name='metric',value_name='value'
    )
    class_fold=(class_long.groupby(['model_id','fraction_percent','class_name','metric','fold'],as_index=False).agg(value=('value','mean')))
    class_summary=(class_fold.groupby(['model_id','fraction_percent','class_name','metric'],as_index=False).agg(
        mean=('value','mean'),sd=('value','std')
    ))
    class_summary['model_name']=class_summary.model_id.map(lambda m: plan.model_metadata.get(m,{}).get('display_name',m))

    primary0,primary1=plan.primary_metrics[:2]
    micro=global_summary[global_summary.metric==primary0].copy()
    macro=global_summary[global_summary.metric==primary1].copy()
    focus_model=next((m for m in models if plan.model_metadata.get(m,{}).get('role')=='relational'),models[0])
    heat_src=class_summary[(class_summary.model_id==focus_model)&(class_summary.metric=='f1')]
    heat=heat_src.pivot(index='class_name',columns='fraction_percent',values='mean').reindex(index=list(dataset.classes))
    heat=heat.reindex(columns=sorted(heat.columns)); heat.columns=[f'{int(c)}%' for c in heat.columns]

    f1_focus=heat_src.pivot(index='class_name',columns='fraction_percent',values='mean')
    low=min(spec['fractions']); high=max(spec['fractions']); low_pct=int(round(low*100)); high_pct=int(round(high*100))
    gain=pd.DataFrame({'class_name':list(dataset.classes)})
    gain=gain.merge(f1_focus[[low_pct,high_pct]].rename(columns={low_pct:'f1_low',high_pct:'f1_high'}).reset_index(),on='class_name',how='left')
    gain['delta_f1']=gain.f1_high-gain.f1_low
    gain['support_full']=dataset.targets.sum(0).astype(int)
    gain['model_id']=focus_model; gain['model_name']=plan.model_metadata.get(focus_model,{}).get('display_name',focus_model)
    gain=gain.sort_values('delta_f1').reset_index(drop=True)

    return {
        'learning_curve':global_summary,
        'learning_curve__micro':micro,
        'learning_curve__macro':macro,
        'learning_curve__by_class':class_summary,
        'learning_curve__class_heatmap':heat,
        'learning_curve__class_gain':gain,
        'learning_curve__subsets':subsets_df,
        'learning_curve__subset_support':support_df,
    }


GRAPH_VIS_CONFIG = {
    'id':'graph_samples',
    'version':'graph_samples_fold_exact_v1',
    'representative_fold_policy':'closest_reference_fold_to_global_primary_metrics',
    'ego_center_policy':'median_validation_mean_cosine_without_labels',
    'ego_second_hop_per_neighbor':2,
    'max_ego_nodes':18,
    'class_policy':'median_global_support_with_validation_presence',
    'class_seed_nodes':6,
    'class_neighbors_per_seed':3,
    'max_class_nodes':24,
    'comparison_policy':'reference_bundle_components_plus_reference',
    'layout_policy':'PCA_on_reference_representation_shared_across_comparison_views',
    'labels_policy':'posthoc_visual_annotation_only',
}


def _reference_condition_for_diagnostics(experiment_result, allowed_representations=None):
    # Mesma heurística usada pelo assembler, aplicada ao contrato de resultados.
    obs=experiment_result.metric_observations
    by_fold=obs.groupby(['condition_id','metric','fold'],as_index=False).value.mean()
    summary=by_fold.groupby(['condition_id','metric'],as_index=False).agg(mean=('value','mean'),sd=('value','std'))
    mean_w=summary.pivot(index='condition_id',columns='metric',values='mean').add_suffix('_mean')
    sd_w=summary.pivot(index='condition_id',columns='metric',values='sd').add_suffix('_sd')
    x=experiment_result.conditions.merge(mean_w.reset_index(),on='condition_id').merge(sd_w.reset_index(),on='condition_id')
    policy=experiment_result.metadata['reference_policy']; metrics=list(policy['metrics'])
    vals=np.column_stack([x[f'{m}_mean'].clip(lower=0).to_numpy(float) for m in metrics])
    denom=np.sum(np.divide(1.0,vals,out=np.full_like(vals,np.inf),where=vals>0),axis=1)
    hmean=np.divide(len(metrics),denom,out=np.zeros(len(x),dtype=float),where=np.isfinite(denom)&(denom>0))
    sd_sum=np.zeros(len(x),dtype=float)
    for m in metrics:
        if f'{m}_sd' in x: sd_sum+=x[f'{m}_sd'].fillna(0).to_numpy(float)
    x['reference_score']=hmean-float(policy.get('stability_weight',0))*sd_sum
    if allowed_representations is not None:
        allowed=set(map(str,allowed_representations))
        x=x[x.representation_name.astype(str).isin(allowed)].copy()
        if x.empty: return {}
    ranking=x.sort_values(['reference_score',*[f'{m}_mean' for m in reversed(metrics)]],ascending=False).reset_index(drop=True)
    return ranking.iloc[0].to_dict()


def _representative_fold_for_condition(experiment_result, condition_id, primary_metrics):
    obs=experiment_result.metric_observations
    q=obs[(obs.condition_id==condition_id)&(obs.metric.isin(primary_metrics))]
    if q.empty: return 0
    fold=q.groupby(['fold','metric'],as_index=False).value.mean().pivot(index='fold',columns='metric',values='value')
    target=q.groupby('metric').value.mean()
    cols=[m for m in primary_metrics if m in fold.columns and m in target.index]
    if not cols: return int(fold.index.min())-1
    dist=np.zeros(len(fold),dtype=float)
    for m in cols: dist+=(fold[m].to_numpy(float)-float(target[m]))**2
    return int(fold.index[int(np.argmin(dist))])-1


def _globalize_graph_edges(edges, tr, va):
    rows=[]
    for r in edges.itertuples(index=False):
        src=int(tr[int(r.origem_treino_local)])
        if str(r.subset)=='validacao': dst=int(va[int(r.destino_local)])
        else: dst=int(tr[int(r.destino_local)])
        rows.append({'source_index':src,'target_index':dst,'rank':int(r.rank),'cosine':float(r.cosseno),'subset':str(r.subset)})
    return pd.DataFrame(rows)


def _features_for_global_indices(pack, bundle, tr, va, global_indices):
    xtr=combine_blocks(pack['train'],bundle); xva=combine_blocks(pack['valid'],bundle)
    tr_map={int(g):i for i,g in enumerate(tr)}; va_map={int(g):i for i,g in enumerate(va)}
    rows=[]
    for g in global_indices:
        g=int(g)
        if g in tr_map: rows.append(xtr[tr_map[g]])
        elif g in va_map: rows.append(xva[va_map[g]])
        else: raise KeyError(f'Índice global fora do fold: {g}')
    return np.asarray(rows,dtype=np.float32)


def _shared_layout(reference_features, center_pos=None):
    x=np.asarray(reference_features,dtype=np.float32)
    if len(x)==1: return np.array([[0.0,0.0]],dtype=np.float32)
    from sklearn.decomposition import PCA
    if np.allclose(x,x[0]):
        ang=np.linspace(0,2*np.pi,len(x),endpoint=False)
        coords=np.column_stack([np.cos(ang),np.sin(ang)]).astype(np.float32)
    else:
        ncomp=min(2,len(x),x.shape[1])
        coords=PCA(n_components=ncomp).fit_transform(x).astype(np.float32)
        if ncomp<2: coords=np.column_stack([coords,np.zeros(len(coords),dtype=np.float32)])
        scale=np.max(np.abs(coords),axis=0); scale[scale<1e-8]=1.0; coords/=scale
    if center_pos is not None:
        coords-=coords[int(center_pos)]
    return coords


def _node_label_names(dataset, idx):
    return [str(c) for c,v in zip(dataset.classes,dataset.targets[int(idx)]) if float(v)>0.5]


def _jaccard_rows(a,b):
    aa=np.asarray(a,dtype=bool); bb=np.asarray(b,dtype=bool)
    union=np.logical_or(aa,bb).sum(); inter=np.logical_and(aa,bb).sum()
    return float(inter/union if union else 1.0)


def _graph_sample_nodes(dataset, indices, tr, va, coords, sample_id, representation_name,
                        center_index=None, focus_class=None, label_indices=()):
    va_set=set(map(int,va)); label_indices=set(map(int,label_indices))
    center_y=dataset.targets[int(center_index)] if center_index is not None else None
    rows=[]
    for pos,g in enumerate(map(int,indices)):
        labels=_node_label_names(dataset,g)
        name=str(dataset.frame.iloc[g].Nome) if 'Nome' in dataset.frame.columns else str(dataset.frame.iloc[g].Id)
        row={
            'sample_id':sample_id,'representation_name':representation_name,'node_index':g,
            'node_id':str(dataset.frame.iloc[g].Id),'display_name':name,'split':'validação' if g in va_set else 'treino',
            'x':float(coords[pos,0]),'y':float(coords[pos,1]),'is_center':bool(center_index is not None and g==int(center_index)),
            'show_label':bool((center_index is not None and g==int(center_index)) or g in label_indices),
            'labels':' | '.join(labels),'n_labels':len(labels),
        }
        if center_y is not None: row['label_jaccard_to_center']=_jaccard_rows(dataset.targets[g],center_y)
        if focus_class is not None:
            j=list(dataset.classes).index(focus_class)
            row['focus_class_present']=bool(dataset.targets[g,j]>0.5)
            row['focus_group']=focus_class if row['focus_class_present'] else 'Outros'
        rows.append(row)
    return pd.DataFrame(rows)


def _sample_edges_for_representation(pack,bundle,tr,va,selected_indices,graph_cfg,sample_id,representation_name):
    xtr=combine_blocks(pack['train'],bundle); xva=combine_blocks(pack['valid'],bundle)
    _,_,edges=build_knn_propagation(xtr,xva,**graph_cfg)
    ge=_globalize_graph_edges(edges,tr,va)
    keep=set(map(int,selected_indices))
    if ge.empty:
        return pd.DataFrame(columns=['sample_id','representation_name','source_index','target_index','rank','cosine','subset'])
    ge=ge[ge.source_index.isin(keep)&ge.target_index.isin(keep)].copy()
    ge.insert(0,'representation_name',representation_name); ge.insert(0,'sample_id',sample_id)
    return ge.reset_index(drop=True)


def run_graph_visualization_diagnostic(dataset, feature_context, experiment_result, plan, spec=GRAPH_VIS_CONFIG):
    if experiment_result is None or experiment_result.conditions.empty:
        return {}
    registry={r['features']:tuple(r['bundle']) for r in plan.registry}
    # Se um experimento estrutural especial vencer o ranking global, a visualização
    # homogênea usa a melhor representação pertencente à matriz principal.
    reference=_reference_condition_for_diagnostics(experiment_result,allowed_representations=registry)
    if not reference: return {}
    ref_name=str(reference['representation_name'])
    ref_bundle=registry[ref_name]
    fold_index=_representative_fold_for_condition(experiment_result,reference['condition_id'],plan.primary_metrics)
    tr,va=dataset.folds[fold_index]; pack=feature_context.fold_packs[fold_index]

    xtr=combine_blocks(pack['train'],ref_bundle); xva=combine_blocks(pack['valid'],ref_bundle)
    _,_,edges_ref=build_knn_propagation(xtr,xva,**plan.graph_config)
    ge_ref=_globalize_graph_edges(edges_ref,tr,va)
    valid_edges=edges_ref[edges_ref.subset=='validacao']
    if valid_edges.empty: return {}

    # Centro do ego-graph: conectividade típica, sem consultar rótulos.
    qmean=valid_edges.groupby('destino_local').cosseno.mean().sort_index()
    med=float(qmean.median()); q=int((qmean-med).abs().sort_values(kind='stable').index[0])
    center=int(va[q])
    first=valid_edges[valid_edges.destino_local==q].sort_values('rank')
    first_globals=[int(tr[int(j)]) for j in first.origem_treino_local]
    selected=[center,*first_globals]
    train_edges=edges_ref[edges_ref.subset=='treino']
    for g in first_globals:
        local=np.where(tr==g)[0]
        if not len(local): continue
        sec=train_edges[train_edges.destino_local==int(local[0])].sort_values('rank').head(int(spec['ego_second_hop_per_neighbor']))
        selected.extend(int(tr[int(j)]) for j in sec.origem_treino_local)
    selected=list(dict.fromkeys(selected))[:int(spec['max_ego_nodes'])]
    first_label=set(first_globals)
    ref_feat=_features_for_global_indices(pack,ref_bundle,tr,va,selected)
    coords=_shared_layout(ref_feat,center_pos=selected.index(center))

    nodes_all=[]; edges_all=[]; sample_rows=[]
    # Referência + componentes individuais do bundle, todos com os mesmos nós/layout.
    comparison=[]
    for src in ref_bundle:
        if src in pack['train']: comparison.append((str(src),(src,)))
    comparison.append((ref_name,ref_bundle))
    seen=set(); comparison=[x for x in comparison if not (x[0] in seen or seen.add(x[0]))]

    for rep_name,bundle in comparison:
        sample_id='graph_ego_reference' if rep_name==ref_name else f"graph_compare__{re.sub(r'[^a-zA-Z0-9_]+','_',rep_name)}"
        nodes=_graph_sample_nodes(dataset,selected,tr,va,coords,sample_id,rep_name,center_index=center,label_indices=first_label)
        ed=_sample_edges_for_representation(pack,bundle,tr,va,selected,plan.graph_config,sample_id,rep_name)
        nodes_all.append(nodes); edges_all.append(ed)
        sample_rows.append({
            'sample_id':sample_id,'view_type':'ego' if rep_name==ref_name else 'comparison',
            'representation_name':rep_name,'fold':fold_index+1,'center_id':str(dataset.frame.iloc[center].Id),
            'center_name':str(dataset.frame.iloc[center].Nome),'focus_class':'',
            'n_nodes':len(nodes),'n_edges':len(ed),
            'title':('Ego-graph' if rep_name==ref_name else 'Mesma amostra')+f' — {rep_name}',
            'color_mode':'center_jaccard','note':'Layout compartilhado da representação de referência; cor = Jaccard de rótulos com o centro (post-hoc).',
        })

    # Subgrafo por classe: classe de suporte mediano com presença no fold representativo.
    support=dataset.targets.sum(0).astype(int); median=float(np.median(support))
    candidates=sorted(range(len(dataset.classes)),key=lambda j:(abs(int(support[j])-median),str(dataset.classes[j])))
    focus_j=next((j for j in candidates if dataset.targets[va,j].sum()>0),candidates[0]); focus_class=str(dataset.classes[focus_j])
    positives=[int(g) for g in va if dataset.targets[int(g),focus_j]>0.5]
    positives=sorted(positives,key=lambda g:str(dataset.frame.iloc[g].Id))[:int(spec['class_seed_nodes'])]
    class_selected=list(positives)
    for g in positives:
        qlocal=int(np.where(va==g)[0][0])
        nbr=valid_edges[valid_edges.destino_local==qlocal].sort_values('rank').head(int(spec['class_neighbors_per_seed']))
        class_selected.extend(int(tr[int(j)]) for j in nbr.origem_treino_local)
    class_selected=list(dict.fromkeys(class_selected))[:int(spec['max_class_nodes'])]
    class_feat=_features_for_global_indices(pack,ref_bundle,tr,va,class_selected)
    class_coords=_shared_layout(class_feat)
    class_id='graph_class_reference'
    class_nodes=_graph_sample_nodes(dataset,class_selected,tr,va,class_coords,class_id,ref_name,focus_class=focus_class,label_indices=positives)
    class_edges=_sample_edges_for_representation(pack,ref_bundle,tr,va,class_selected,plan.graph_config,class_id,ref_name)
    nodes_all.append(class_nodes); edges_all.append(class_edges)
    sample_rows.append({
        'sample_id':class_id,'view_type':'class','representation_name':ref_name,'fold':fold_index+1,
        'center_id':'','center_name':'','focus_class':focus_class,'n_nodes':len(class_nodes),'n_edges':len(class_edges),
        'title':f'Subgrafo por macrogênero — {focus_class} — {ref_name}',
        'color_mode':'focus_class','note':'Seeds de validação da classe + vizinhos de treino; rótulos usados apenas para seleção/coloração post-hoc.',
    })

    nodes_df=pd.concat(nodes_all,ignore_index=True) if nodes_all else pd.DataFrame()
    edges_df=pd.concat(edges_all,ignore_index=True) if edges_all else pd.DataFrame()
    samples_df=pd.DataFrame(sample_rows)
    return {
        'graph_samples__samples':samples_df,
        'graph_samples__nodes':nodes_df,
        'graph_samples__edges':edges_df,
    }


def _focused_condition_rows(experiment_result, wanted):
    cond=experiment_result.conditions; obs=experiment_result.metric_observations
    rows=[]
    for representation,model,label,role in wanted:
        q=cond[(cond.representation_name==representation)&(cond.model_id==model)]
        if q.empty: continue
        cid=str(q.iloc[0].condition_id); o=obs[obs.condition_id==cid]
        row={'condition_id':cid,'condition_label':label,'comparison_role':role,'representation_name':representation,'model_id':model}
        for metric in ('f1_micro','f1_macro'):
            vals=o[o.metric==metric].groupby('fold').value.mean()
            row[f'{metric}_mean']=float(vals.mean()) if len(vals) else np.nan
            row[f'{metric}_sd']=float(vals.std()) if len(vals)>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_graph_operator_diagnostic(experiment_result):
    """Mesma representação e mesmo kNN; muda apenas o operador de message passing."""
    wanted=[
        ('minilm+word_1','mlp','MLP','individual_vector'),
        ('minilm+word_1','gnn','Fixed-kNN GNN','fixed_message_passing'),
        ('minilm+word_1','gcn','GCN','gcn'),
        ('minilm+word_1','graphsage','GraphSAGE','graphsage'),
        ('minilm+word_1','gatv2','GATv2','attention'),
    ]
    comparison=_focused_condition_rows(experiment_result,wanted)
    if not comparison.empty:
        base=comparison[comparison.model_id=='mlp']
        if not base.empty:
            for metric in ('f1_micro','f1_macro'):
                comparison[f'delta_{metric}_vs_mlp']=comparison[f'{metric}_mean']-float(base.iloc[0][f'{metric}_mean'])
    return {'graph_operator_comparison':comparison}


def run_hetero_structural_diagnostic(experiment_result):
    """Vetor, grafo homogêneo e ablações heterogêneas da hipótese de difusão."""
    base=HETERO_TERM_CFG['representation_name']
    wanted=[
        ('minilm+word_1','mlp','MLP — MiniLM+word_1','individual_vector'),
        ('minilm+word_1','gnn','Fixed-kNN GNN — MiniLM+word_1','homogeneous_graph'),
        (base,'hetero_term_gnn','Hetero — todos os termos','heterogeneous_all_terms'),
        (f'{base}_top30','hetero_term_gnn','Hetero — top-30 TF-IDF','heterogeneous_topk'),
        (f'{base}_top20','hetero_term_gnn','Hetero — top-20 TF-IDF','heterogeneous_topk'),
        (f'{base}_top10','hetero_term_gnn','Hetero — top-10 TF-IDF','heterogeneous_topk'),
        (base,'hetero_term_gatv2','Hetero GATv2 — todos os termos','heterogeneous_attention'),
    ]
    comparison=_focused_condition_rows(experiment_result,wanted)
    if not comparison.empty:
        base_row=comparison[comparison.comparison_role=='individual_vector']
        if not base_row.empty:
            for metric in ('f1_micro','f1_macro'):
                b=float(base_row.iloc[0][f'{metric}_mean'])
                comparison[f'delta_{metric}_vs_mlp']=comparison[f'{metric}_mean']-b

    raw=experiment_result.raw_metrics
    structure_cols=['features','modelo','fold','term_nodes','incidence_train','incidence_valid','zero_term_valid_rows','term_top_k']
    structure=pd.DataFrame()
    if not raw.empty and all(c in raw.columns for c in structure_cols):
        q=raw[raw.modelo.isin(['hetero_term_gnn','hetero_term_gatv2'])].copy()
        if not q.empty:
            structure=q[structure_cols+['n_treino']].drop_duplicates().sort_values(['features','modelo','fold']).reset_index(drop=True)
            structure['mean_terms_per_train_work']=structure.incidence_train/np.maximum(1,structure.n_treino)
            structure=structure.drop(columns='n_treino')
    return {'hetero_structural_comparison':comparison,'hetero_term_structure':structure}


def run_diagnostics(dataset, feature_context, experiment_result, plan):
    diagnostics={}
    diagnostics.update(run_relational_diagnostics(dataset,feature_context,experiment_result,plan))
    diagnostics.update(run_graph_operator_diagnostic(experiment_result))
    diagnostics.update(run_hetero_structural_diagnostic(experiment_result))
    diagnostics.update(run_learning_curve_diagnostic(dataset,feature_context,experiment_result,plan))
    diagnostics.update(run_graph_visualization_diagnostic(dataset,feature_context,experiment_result,plan))
    return diagnostics
