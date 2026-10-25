"""Experimental study protocol, model metadata, and environment paths.

Central configuration for:
- Cross-validation protocol and random seeds
- Enabled models, metrics, and reference policies
- Graph topology and inductive operator settings
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
NEURAL_SEEDS = (42, 43, 44)
CLASSICAL_SEEDS = (42,)

MODEL_SEEDS: dict[str, tuple[int, ...]] = {
    "linear": NEURAL_SEEDS,
    "mlp": NEURAL_SEEDS,
    "rf": CLASSICAL_SEEDS,
    "xgb": CLASSICAL_SEEDS,
    "gnn": NEURAL_SEEDS,
    "gcn": NEURAL_SEEDS,
    "graphsage": NEURAL_SEEDS,
    "gatv2": NEURAL_SEEDS,
    "hetero_term_gnn": NEURAL_SEEDS,
    "hetero_term_gatv2": NEURAL_SEEDS,
}

MODEL_ENABLED: dict[str, bool] = {
    "linear": True,
    "mlp": True,
    "rf": False,
    "xgb": False,
    "gnn": True,
    "gcn": True,              # Focal experiment on MiniLM+word_1
    "graphsage": True,        # Focal experiment on MiniLM+word_1
    "gatv2": True,            # Focal experiment on MiniLM+word_1
    "hetero_term_gnn": True,  # Structural bipartite experiment
    "hetero_term_gatv2": True,
}

MAIN_MODEL_ORDER = ("linear", "mlp", "rf", "xgb", "gnn")
MODELS = tuple(m for m in MAIN_MODEL_ORDER if MODEL_ENABLED.get(m, False))

MODEL_METADATA: dict[str, dict[str, Any]] = {
    "linear": {
        "display_name": "Linear",
        "family": "linear",
        "role": "individual",
        "enabled": True,
        "execution_scope": "main_matrix",
    },
    "mlp": {
        "display_name": "MLP",
        "family": "neural",
        "role": "individual_baseline",
        "enabled": True,
        "execution_scope": "main_matrix",
    },
    "rf": {
        "display_name": "Random Forest",
        "family": "tree",
        "role": "individual",
        "enabled": False,
        "execution_scope": "main_matrix",
    },
    "xgb": {
        "display_name": "XGBoost",
        "family": "boosting",
        "role": "individual",
        "enabled": False,
        "execution_scope": "main_matrix",
    },
    "gnn": {
        "display_name": "Fixed-kNN GNN",
        "family": "graph",
        "role": "relational",
        "enabled": True,
        "execution_scope": "main_matrix",
    },
    "gcn": {
        "display_name": "GCN",
        "family": "graph",
        "role": "relational_operator",
        "enabled": True,
        "execution_scope": "special_experiment",
    },
    "graphsage": {
        "display_name": "GraphSAGE",
        "family": "graph",
        "role": "relational_operator",
        "enabled": True,
        "execution_scope": "special_experiment",
    },
    "gatv2": {
        "display_name": "GATv2",
        "family": "graph_attention",
        "role": "relational_operator",
        "enabled": True,
        "execution_scope": "special_experiment",
    },
    "hetero_term_gnn": {
        "display_name": "Hetero GNN (obra↔termo)",
        "family": "heterogeneous_graph",
        "role": "relational_structural",
        "enabled": True,
        "execution_scope": "special_experiment",
    },
    "hetero_term_gatv2": {
        "display_name": "Hetero GATv2 (termo→obra)",
        "family": "heterogeneous_graph_attention",
        "role": "relational_structural",
        "enabled": True,
        "execution_scope": "special_experiment",
    },
}

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
        "relational_model": "gnn",
        "baseline_model": "mlp",
        "metric": "f1_macro",
        "neighbor_quality_metric": "mean_jaccard",
        "display_name": "Homofilia × ganho relacional",
    },
)

RESULT_SCHEMA_VERSION = "study_result_v1"
EXPERIMENT_SCHEMA_VERSION = "experiment_result_v1"

# Hyperparameters
EPOCHS = 200
HIDDEN = 64
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
FIXED_THRESHOLD = 0.5
GRAPH_CFG: dict[str, Any] = {"k": 5, "self_weight": 0.90, "min_similarity": None}

HOMOGENEOUS_OPERATOR_CFG: dict[str, Any] = {
    "representation_name": "minilm+word_1",
    "bundle": ("minilm", "word_1"),
    "models": ("gcn", "graphsage", "gatv2"),
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
    "hidden": HIDDEN,
    "dropout": DROPOUT,
    "inductive_policy": (
        "term states are built from training works only; "
        "validation works never update shared term nodes"
    ),
}

RELATIONAL_RESEARCH_REGISTRY: dict[str, dict[str, str]] = {
    "fixed_knn_gnn": {
        "status": "active_baseline",
        "question": "A propagação kNN fixa melhora sobre o baseline individual?",
        "structure": "directed train-only kNN; fixed 0.9 self + 0.1 uniform neighbor mean",
    },
    "gcn_knn": {
        "status": "active_focal",
        "question": "A normalização convolucional de GCN melhora sobre a propagação fixa no mesmo kNN?",
        "structure": "same directed inductive kNN edges; PyG GCNConv",
    },
    "graphsage_knn": {
        "status": "active_focal",
        "question": "Separar self e média dos vizinhos via GraphSAGE melhora no mesmo kNN?",
        "structure": "same directed inductive kNN edges; PyG SAGEConv",
    },
    "gatv2_knn": {
        "status": "active_focal",
        "question": "Aprender quanto confiar em cada vizinho melhora no mesmo kNN?",
        "structure": "same directed inductive kNN edges; PyG GATv2Conv",
    },
    "hetero_term_gnn": {
        "status": "active_baseline",
        "question": "Representar termos como nós explícitos melhora em relação a concatenar o sinal lexical?",
        "structure": "work nodes + term nodes; weighted TF-IDF incidence; MiniLM on work nodes",
    },
    "hetero_term_topk": {
        "status": "active_focal",
        "question": "Reduzir o número de termos por obra melhora a heterogênea?",
        "structure": "same heterogeneous model with per-work TF-IDF top-k pruning",
    },
    "hetero_term_gatv2": {
        "status": "active_focal",
        "question": "Atenção aprendida termo→obra consegue suprimir relações lexicais pouco úteis?",
        "structure": "term states built from training works only; GATv2 attention on term→work aggregation",
    },
    "hypergraph_terms": {
        "status": "planned_disabled",
        "question": "Relações de alta ordem termo→múltiplas obras são melhores que arestas obra↔obra?",
        "structure": "works as vertices; shared terms/topics as hyperedges",
    },
    "hypergraph_attention": {
        "status": "planned_disabled",
        "question": "Atenção dentro de hiperarestas melhora a seleção de sinal compartilhado?",
        "structure": "hypergraph convolution with attention; only after the non-attentive hypergraph baseline",
    },
}

# ---------------- Environment and Device ----------------
CPU_THREADS = max(1, min(4, os.cpu_count() or 1))

# Framework-agnostic default device string
DEVICE_NAME = os.environ.get("GNN_DEVICE", "cpu")

try:
    import torch
    DEVICE = torch.device(DEVICE_NAME)
    torch.set_num_threads(CPU_THREADS)
except ImportError:
    DEVICE = DEVICE_NAME

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
    """Capture execution platform, hardware threads, and framework environment."""
    ram_gb = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            parts = f.readline().split()
        if len(parts) >= 2:
            ram_gb = float(parts[1]) / 1024 / 1024
    except Exception:
        pass

    torch_ver = None
    cuda_available = False
    try:
        import torch
        torch_ver = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        pass

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_threads_used": CPU_THREADS,
        "ram_gb": ram_gb,
        "torch": torch_ver,
        "cuda_available_but_not_used": cuda_available,
        "device_for_study": DEVICE_NAME,
    }

RUNTIME_INFO = collect_runtime_info()
