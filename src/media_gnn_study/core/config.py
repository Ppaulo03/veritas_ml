"""Experimental study protocol, graph topology rules, and environment paths.

Completely model-agnostic and framework-agnostic configuration:
- Cross-validation protocol and default seeds
- Evaluation metrics and reference selection policies
- Inductive graph construction parameters
- Dataset paths, cache directories, and SHA-256 integrity fingerprints
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

# ---------------- Experimental Protocol ----------------
STUDY_VERSION = "unified_representation_graph_study_v5"
CV_SPLITS = 5
CV_SEED = 42
DEFAULT_MODEL_SEEDS: tuple[int, ...] = (42, 43, 44)

PRIMARY_METRICS = ("f1_micro", "f1_macro")
REPORTED_METRICS = ("f1_micro", "f1_macro", "precision_micro", "recall_micro")
CLASS_METRICS = ("precision", "recall", "f1", "support")
METRIC_LABELS: dict[str, str] = {
    "f1_micro": "Micro-F1",
    "f1_macro": "Macro-F1",
    "precision_micro": "Precisão micro",
    "recall_micro": "Recall micro",
    "precision": "Precisão",
    "recall": "Recall",
    "f1": "F1",
    "support": "Support",
}

REFERENCE_POLICY: dict[str, Any] = {
    "name": "harmonic_primary_with_stability_penalty_v1",
    "metrics": PRIMARY_METRICS,
    "stability_weight": 0.10,
    "time_tiebreak_component": "pipeline_incremental",
}

RELATIONAL_DIAGNOSTICS: tuple[dict[str, Any], ...] = (
    {
        "id": "relational_vs_individual",
        "relational_role": "relational",
        "baseline_role": "individual_baseline",
        "metric": "f1_macro",
        "neighbor_quality_metric": "mean_jaccard",
        "display_name": "Homofilia × ganho relacional",
    },
)

RESULT_SCHEMA_VERSION = "study_result_v1"
EXPERIMENT_SCHEMA_VERSION = "experiment_result_v1"
FIXED_THRESHOLD = 0.5

# ---------------- Graph Topology Rules ----------------
GRAPH_CFG: dict[str, Any] = {"k": 5, "self_weight": 0.90, "min_similarity": None}

HOMOGENEOUS_OPERATOR_CFG: dict[str, Any] = {
    "representation_name": "minilm+word_1",
    "bundle": ("minilm", "word_1"),
    "k": GRAPH_CFG["k"],
    "inductive_policy": (
        "validation nodes receive messages only from training nodes; "
        "no validation→training or validation→validation edges"
    ),
}

HETERO_DIFFUSION_CFG: dict[str, Any] = {
    "top_k_values": (30, 20, 10),
    "attention_relation": "term_to_work_only",
    "question": "Does pruning or learned attention reduce diffuse lexical signal in the work↔term graph?",
}

HETERO_TERM_CFG: dict[str, Any] = {
    "id": "hetero_term_gnn_v1",
    "representation_name": "hetero_minilm+term_nodes",
    "work_source": "minilm",
    "term_source": "word_1",
    "term_weight": "tfidf_train_only",
    "inductive_policy": (
        "term states are built from training works only; "
        "validation works never update shared term nodes"
    ),
}

# ---------------- Environment and Dataset Constants ----------------
CPU_THREADS = max(1, min(4, os.cpu_count() or 1))
DEVICE_NAME = os.environ.get("GNN_DEVICE", "cpu")

TARGET = "Macrogênero (Gênero unificado)"
CLASSES = [
    "Aventura", "Ação", "Comédia", "Cozy", "Drama", "Esportes", "Fantasia",
    "Ficção Científica", "Guerra", "Mistério", "Musical", "Realidade", "Romance", "Terror"
]

DATA_SHA = "55808c037dd84fec48b9f272d2187e6372590ad0b4f083f0acc831695e7c5907"
SPLIT_SHA = "839157ec5a84b8f6542adda81122602529012815fc73bb9e78ddff8eb6a3ed38"
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EMBED_RULE = "token_chunks_no_overlap_content_weighted_mean_l2_v1"
REFERENCE_MINILM_RUN = "20261021T002245Z_227139"
SPACY_MODEL_ID = "pt_core_news_sm"
EMOTION_MODEL_ID = "MilaNLProc/xlm-emo-t"
ENTITY_HASH_DIM = 64
LEX_SVD_DIM = 128
ACTION_SVD_DIM = 64
NMF_DIM = 16

# ---------------- Paths ----------------
ROOT = Path(os.environ.get("GNN_PROJECT_ROOT", Path.cwd())).expanduser().resolve()
DATA = ROOT / "01_dados" / "medias_classificadas.csv"
PARTS = ROOT / "01_dados" / "particoes_franquias.csv"
LEGACY_PARTS = ROOT / "03_resultados" / "01_1_qualidade_extracao" / "20261019T230939Z" / "particoes_franquias.csv"
CENTRAL = ROOT / "03_resultados" / "00_estudo_central"
CACHE = CENTRAL / "cache"
RUN_ID = os.environ.get("GNN_RUN_ID") or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6])
RUN_DIR = CENTRAL / "runs" / RUN_ID
STUDY_DIR = RUN_DIR / "estudo_unificado"

INCREMENTAL_REUSE = os.getenv("GNN_INCREMENTAL_REUSE", "1") == "1"
INCREMENTAL_REUSE_STRICT = os.getenv("GNN_INCREMENTAL_REUSE_STRICT", "1") == "1"
INCREMENTAL_BASE_RUN_ID = os.getenv("GNN_INCREMENTAL_BASE_RUN_ID", "20261025T142618Z_d9aa19")


def ensure_runtime_dirs() -> Path:
    """Ensure required output and cache directories exist on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    if not PARTS.is_file() and LEGACY_PARTS.is_file():
        PARTS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LEGACY_PARTS, PARTS)
    return STUDY_DIR


def collect_runtime_info() -> dict[str, Any]:
    """Capture execution platform and hardware environment."""
    ram_gb = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            parts = f.readline().split()
        if len(parts) >= 2:
            ram_gb = float(parts[1]) / 1024 / 1024
    except Exception:
        pass

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_threads_used": CPU_THREADS,
        "ram_gb": ram_gb,
        "device_for_study": DEVICE_NAME,
    }


RUNTIME_INFO = collect_runtime_info()
