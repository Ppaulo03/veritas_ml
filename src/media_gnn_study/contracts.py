
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

from dataclasses import dataclass, field
from .config import *

@dataclass(frozen=True)
class DatasetContext:
    frame: pd.DataFrame
    targets: np.ndarray
    classes: tuple
    folds: tuple
    fold_id: np.ndarray
    partitions: pd.DataFrame
    metadata: dict = field(default_factory=dict)


@dataclass
class FeatureContext:
    fold_packs: list
    catalog: pd.DataFrame
    fixed_meta: dict
    fixed_extraction_timing: pd.DataFrame
    fold_meta: pd.DataFrame
    component_timing: pd.DataFrame


@dataclass(frozen=True)
class ExperimentPlan:
    registry: tuple
    model_ids: tuple
    model_seeds: dict
    model_metadata: dict
    primary_metrics: tuple
    reported_metrics: tuple
    class_metrics: tuple
    metric_labels: dict
    reference_policy: dict
    graph_config: dict
    diagnostics: tuple = ()
    special_experiments: tuple = ()
    registered_models: tuple = ()


@dataclass
class ExperimentResult:
    schema_version: str
    conditions: pd.DataFrame
    metric_observations: pd.DataFrame
    class_metric_observations: pd.DataFrame
    timing_observations: pd.DataFrame
    raw_metrics: pd.DataFrame
    raw_class_metrics: pd.DataFrame
    metadata: dict = field(default_factory=dict)


@dataclass
class StudyResult:
    schema_version: str
    metadata: dict
    tables: dict
    reference: dict
    table_specs: list
    plot_specs: list


def stable_id(prefix, value):
    digest = hashlib.sha1(str(value).encode('utf-8')).hexdigest()[:12]
    return f'{prefix}_{digest}'
