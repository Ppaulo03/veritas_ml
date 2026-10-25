
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

import matplotlib.pyplot as plt
from .config import *
from .contracts import *

def validate_study_result(result):
    if not isinstance(result,StudyResult):
        raise TypeError('render_study espera StudyResult.')
    if not isinstance(result.tables,dict) or not isinstance(result.plot_specs,list):
        raise ValueError('StudyResult inválido.')
    for spec in result.table_specs:
        if spec['table'] not in result.tables:
            raise KeyError(f"Tabela ausente: {spec['table']}")
    for spec in result.plot_specs:
        if spec['table'] not in result.tables:
            raise KeyError(f"Tabela de gráfico ausente: {spec['table']}")
        if spec.get('edges_table') and spec['edges_table'] not in result.tables:
            raise KeyError(f"Tabela de arestas ausente: {spec['edges_table']}")
    return True


def _save_rendered_figure(fig,out_dir,plot_id,dpi=160):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    path=out_dir/f'{plot_id}.png'; fig.tight_layout(); fig.savefig(path,dpi=dpi,bbox_inches='tight')
    plt.show(); plt.close(fig); return path


def _auto_limits(values,bounds=None,pad=0.08,log=False):
    arr=np.asarray(values,dtype=float); arr=arr[np.isfinite(arr)]
    if log: arr=arr[arr>0]
    if len(arr)==0: return None
    lo=float(arr.min()); hi=float(arr.max())
    if log:
        if lo==hi: lo*=0.8; hi*=1.2
        else: lo/=1.15; hi*=1.15
    else:
        span=hi-lo
        if span<1e-12: span=max(abs(hi)*0.1,0.05)
        lo-=pad*span; hi+=pad*span
    if bounds is not None:
        lo=max(float(bounds[0]),lo); hi=min(float(bounds[1]),hi)
    if hi<=lo: hi=lo+1e-6
    return lo,hi


def _label_subset(df,spec):
    filters=spec.get('label_filters') or []
    if filters:
        mask=np.zeros(len(df),dtype=bool)
        for col in filters:
            if col in df: mask |= df[col].fillna(False).astype(bool).to_numpy()
        return df.loc[mask]
    highlight=spec.get('highlight')
    if highlight and highlight in df and df[highlight].fillna(False).any():
        return df[df[highlight].fillna(False)]
    if len(df)>20:
        return pd.concat([df.nlargest(4,spec['y']),df.nsmallest(4,spec['y'])]).drop_duplicates()
    return df


def _add_trend_and_correlation(ax,df,spec):
    if not (spec.get('trendline') or spec.get('correlation')):
        return
    x=pd.to_numeric(df[spec['x']],errors='coerce').to_numpy(float)
    y=pd.to_numeric(df[spec['y']],errors='coerce').to_numpy(float)
    mask=np.isfinite(x)&np.isfinite(y)
    x=x[mask]; y=y[mask]
    if len(x)<2 or np.std(x)<1e-12 or np.std(y)<1e-12:
        return
    if spec.get('trendline'):
        coef=np.polyfit(x,y,1); grid=np.linspace(x.min(),x.max(),100)
        ax.plot(grid,coef[0]*grid+coef[1],linestyle='--',linewidth=1.3,label='Tendência linear')
    if spec.get('correlation'):
        r=float(np.corrcoef(x,y)[0,1])
        ax.text(.02,.98,f'Pearson r = {r:.2f}',transform=ax.transAxes,va='top',ha='left',fontsize=9,
                bbox={'boxstyle':'round','facecolor':'white','alpha':.75,'edgecolor':'none'})


def _filter_table(df, filters):
    out=df
    for col,val in (filters or {}).items():
        if col in out.columns: out=out[out[col].astype(str)==str(val)]
    return out.copy()


def _render_network(spec,tables,out_dir):
    nodes=_filter_table(tables[spec['table']],spec.get('filters'))
    edges=_filter_table(tables[spec['edges_table']],spec.get('filters')) if spec.get('edges_table') else pd.DataFrame()
    if nodes.empty:
        raise ValueError(f"Amostra de grafo vazia: {spec['id']}")
    fig,ax=plt.subplots(figsize=(9,7))
    pos={int(r.node_index):(float(r.x),float(r.y)) for r in nodes.itertuples(index=False)}

    if not edges.empty:
        wcol=spec.get('edge_weight')
        vals=pd.to_numeric(edges[wcol],errors='coerce').fillna(0).to_numpy(float) if wcol and wcol in edges else np.ones(len(edges))
        if len(vals):
            lo,hi=float(np.nanmin(vals)),float(np.nanmax(vals)); span=max(hi-lo,1e-9)
        else: lo=0.0; span=1.0
        for r,w in zip(edges.itertuples(index=False),vals):
            s=int(r.source_index); t=int(r.target_index)
            if s not in pos or t not in pos: continue
            x0,y0=pos[s]; x1,y1=pos[t]
            width=.45+1.8*max(0.0,min(1.0,(float(w)-lo)/span))
            ax.plot([x0,x1],[y0,y1],linewidth=width,alpha=.28,color='0.55',zorder=1)

    color_numeric=spec.get('color_numeric')
    group=spec.get('group')
    if color_numeric and color_numeric in nodes:
        from matplotlib.colors import Normalize
        vals=pd.to_numeric(nodes[color_numeric],errors='coerce').fillna(0).to_numpy(float)
        norm=Normalize(vmin=float(np.nanmin(vals)),vmax=float(np.nanmax(vals)) if np.nanmax(vals)>np.nanmin(vals) else float(np.nanmin(vals))+1e-6)
        sc=ax.scatter(nodes.x,nodes.y,c=vals,cmap='viridis',norm=norm,s=90,edgecolors='white',linewidths=.8,zorder=3)
        fig.colorbar(sc,ax=ax,label=spec.get('colorbar',color_numeric))
    elif group and group in nodes:
        for name,g in nodes.groupby(group,sort=True):
            ax.scatter(g.x,g.y,s=90,edgecolors='white',linewidths=.8,label=str(name),zorder=3)
    else:
        ax.scatter(nodes.x,nodes.y,s=90,edgecolors='white',linewidths=.8,zorder=3)

    center_col=spec.get('center')
    if center_col and center_col in nodes and nodes[center_col].fillna(False).any():
        c=nodes[nodes[center_col].fillna(False)]
        ax.scatter(c.x,c.y,marker='*',s=320,facecolors='none',edgecolors='black',linewidths=1.7,label='Centro',zorder=5)

    label_col=spec.get('label'); flag=spec.get('label_flag')
    if label_col and label_col in nodes:
        lab=nodes[nodes[flag].fillna(False)] if flag and flag in nodes else nodes.head(12)
        for r in lab.itertuples(index=False):
            txt=str(getattr(r,label_col)); txt=txt if len(txt)<=28 else txt[:25]+'…'
            ax.annotate(txt,(float(r.x),float(r.y)),xytext=(5,5),textcoords='offset points',fontsize=7,zorder=6)

    title=spec.get('title','')
    subtitle=spec.get('subtitle')
    ax.set_title(title+(f'\n{subtitle}' if subtitle else ''))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('')
    ax.set_aspect('equal',adjustable='datalim')
    for spine in ax.spines.values(): spine.set_visible(False)
    handles,labels=ax.get_legend_handles_labels()
    if handles: ax.legend(fontsize=8,loc='best')
    return _save_rendered_figure(fig,out_dir,spec['id'])


def _render_plot_spec(spec,tables,out_dir):
    df=_filter_table(tables[spec['table']],spec.get('filters'))
    kind=spec['kind']; fig=None
    if kind=='network':
        return _render_network(spec,tables,out_dir)
    if kind in ('bar','barh'):
        fig,ax=plt.subplots(figsize=(9,6))
        if kind=='barh': ax.barh(df[spec['y']],df[spec['x']])
        else: ax.bar(df[spec['x']].astype(str),df[spec['y']])
    elif kind=='grouped_barh':
        fig,ax=plt.subplots(figsize=(10,max(5,0.36*len(df)+2)))
        y=np.arange(len(df)); series=spec['series']; labels=spec.get('series_labels',series); h=.75/max(len(series),1)
        errors=spec.get('errors',[None]*len(series)); offsets=(np.arange(len(series))-(len(series)-1)/2)*h
        for off,col,label,err_col in zip(offsets,series,labels,errors):
            xerr=df[err_col] if err_col and err_col in df else None
            ax.barh(y+off,df[col],height=h,label=label,xerr=xerr,capsize=2 if xerr is not None else 0)
        ax.set_yticks(y,df[spec['y']])
    elif kind=='scatter':
        fig,ax=plt.subplots(figsize=(8,6))
        group=spec.get('group')
        if group and group in df:
            for name,g in df.groupby(group): ax.scatter(g[spec['x']],g[spec['y']],label=str(name),alpha=.65)
        else: ax.scatter(df[spec['x']],df[spec['y']],alpha=.7)
        label_col=spec.get('label')
        if label_col and label_col in df:
            label_df=_label_subset(df,spec)
            for row in label_df.itertuples(index=False):
                ax.annotate(str(getattr(row,label_col)),(getattr(row,spec['x']),getattr(row,spec['y'])),xytext=(4,4),textcoords='offset points',fontsize=7)
        highlight=spec.get('highlight')
        if highlight and highlight in df and df[highlight].fillna(False).any():
            hdf=df[df[highlight].fillna(False)]
            ax.scatter(hdf[spec['x']],hdf[spec['y']],marker='*',s=220,label='Referência')
        line_filter=spec.get('line_filter')
        if line_filter and line_filter in df and df[line_filter].fillna(False).any():
            ldf=df[df[line_filter].fillna(False)].sort_values(spec['x'])
            ax.plot(ldf[spec['x']],ldf[spec['y']],marker='o',label='Pareto')
        _add_trend_and_correlation(ax,df,spec)
    elif kind=='line_error':
        fig,ax=plt.subplots(figsize=(8,6))
        group=spec.get('group')
        groups=df.groupby(group) if group and group in df else [('Série',df)]
        for name,g in groups:
            g=g.sort_values(spec['x']); err=g[spec['error']] if spec.get('error') in g else None
            ax.errorbar(g[spec['x']],g[spec['y']],yerr=err,marker='o',capsize=3,label=str(name))
        if spec.get('xtick_percent'):
            ticks=sorted(pd.unique(df[spec['x']].dropna()))
            ax.set_xticks(ticks,[f'{int(t)}%' for t in ticks])
    elif kind=='heatmap':
        fig,ax=plt.subplots(figsize=(9,8)); arr=df.to_numpy(float)
        im=ax.imshow(arr,aspect='auto',vmin=spec.get('vmin'),vmax=spec.get('vmax'))
        ax.set_xticks(range(len(df.columns)),[str(x) for x in df.columns],rotation=90)
        ax.set_yticks(range(len(df.index)),[str(x) for x in df.index])
        if spec.get('annotate'):
            fmt=spec.get('fmt','.0f')
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if np.isfinite(arr[i,j]): ax.text(j,i,format(arr[i,j],fmt),ha='center',va='center',fontsize=7)
        fig.colorbar(im,ax=ax,label=spec.get('colorbar',''))
    else:
        raise ValueError(f'Tipo de gráfico não suportado: {kind}')

    if spec.get('xscale'): ax.set_xscale(spec['xscale'])
    if spec.get('yscale'): ax.set_yscale(spec['yscale'])
    ax.set_title(spec.get('title',''))
    if spec.get('xlabel'): ax.set_xlabel(spec['xlabel'])
    if spec.get('ylabel'): ax.set_ylabel(spec['ylabel'])
    if spec.get('xlim'): ax.set_xlim(*spec['xlim'])
    elif spec.get('auto_xlim') and spec.get('x') in df:
        lim=_auto_limits(df[spec['x']],spec.get('x_bounds'),log=spec.get('xscale')=='log')
        if lim: ax.set_xlim(*lim)
    if spec.get('ylim'): ax.set_ylim(*spec['ylim'])
    elif spec.get('auto_ylim') and spec.get('y') in df:
        lim=_auto_limits(df[spec['y']],spec.get('y_bounds'),log=spec.get('yscale')=='log')
        if lim: ax.set_ylim(*lim)
    if 'hline' in spec: ax.axhline(spec['hline'],linewidth=1)
    if 'vline' in spec: ax.axvline(spec['vline'],linewidth=1)
    handles,labels=ax.get_legend_handles_labels()
    if handles: ax.legend(fontsize=8)
    return _save_rendered_figure(fig,out_dir,spec['id'])


def persist_study_result(result,directory):
    validate_study_result(result)
    directory=Path(directory); table_dir=directory/'tables'; table_dir.mkdir(parents=True,exist_ok=True)
    for name,df in result.tables.items():
        if isinstance(df,pd.DataFrame): df.to_csv(table_dir/f'{name}.csv',index=True if df.index.name or not isinstance(df.index,pd.RangeIndex) else False)
    meta={**result.metadata,'schema_version':result.schema_version,'reference':result.reference,'tables':sorted(result.tables),'plots':[p['id'] for p in result.plot_specs]}
    (directory/'study_result.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False,default=str),encoding='utf-8')


def render_study(result,output_dir):
    validate_study_result(result)
    print('=== STUDY RESULT ===')
    print('Schema:',result.schema_version)
    print('Referência exploratória:',result.reference.get('representation_name'),'×',result.reference.get('model_name'))
    for spec in result.table_specs:
        print(f"\n=== {spec['title']} ===")
        df=result.tables[spec['table']]
        display(df.head(spec['limit']) if spec.get('limit') else df)
    fig_dir=Path(output_dir)/'figures'; paths=[]
    for spec in result.plot_specs:
        paths.append(_render_plot_spec(spec,result.tables,fig_dir))
    print(f'\n{len(paths)} figuras salvas em: {fig_dir}')
    return paths
