# Media GNN Study

Framework experimental 100% agnóstico a modelos para classificação multirrótulo de obras de mídia e benchmark relacional em grafos.

- **Framework (`src/media_gnn_study/`):** Contratos imutáveis, protocolo indutivo (zero vazamento validação→treino), álgebra de grafos em NumPy/SciPy, extração de representações, motor de avaliação e renderizador declarativo.
- **Notebook (`notebooks/01_estudo_local.ipynb`):** Definição das arquiteturas de modelos (PyTorch, PyG, Scikit-Learn/XGBoost ou TensorFlow/Keras), hiperparâmetros de treino e registro no framework.

## Estrutura do Repositório

```text
├── notebooks/
│   └── 01_estudo_local.ipynb   # Arquiteturas de modelos, registro e execução local
├── src/media_gnn_study/
│   ├── core/
│   │   ├── config.py           # Protocolo experimental e caminhos (sem modelos acoplados)
│   │   ├── contracts.py        # DatasetContext / FeatureContext / GraphContext / ExperimentResult / StudyResult
│   │   └── graph.py            # Álgebra indutiva de grafos pura em NumPy/SciPy (kNN, poda top-k, incidência bipartida)
│   ├── data/
│   │   ├── loader.py           # Leitura do CSV, limpeza e validação criptográfica SHA-256
│   │   └── splits.py           # Folds agrupados por franquia sem vazamento
│   ├── features/
│   │   ├── encoders.py         # MiniLM e XLM-RoBERTa emoção (com cache em disco)
│   │   ├── linguistic.py       # spaCy NER, pares verbo-objeto e métricas sintáticas
│   │   └── lexical.py          # TF-IDF, SVD, NMF e montagem fold-local do FeatureContext
│   ├── models/
│   │   ├── base.py             # Contratos ModelOutput, ModelSpec e protocolo ModelAdapterFn
│   │   ├── metrics.py          # Métricas multirrótulo e por classe
│   │   └── registry.py         # Registro agnóstico (register_model / register_special_experiment / run_model)
│   ├── engine/
│   │   ├── plan.py             # Catálogo de 55 bundles e montagem dinâmica do ExperimentPlan
│   │   ├── sanity.py           # Verificações pré-execução do protocolo indutivo
│   │   └── runner.py           # Execução incremental agnóstica a framework
│   ├── evaluation/
│   │   ├── diagnostics.py      # Homofilia × ganho, learning curves e subgrafos qualitativos
│   │   ├── study.py            # Montagem do StudyResult e fronteira de Pareto
│   │   ├── render.py           # Renderizador declarativo de tabelas e figuras
│   │   └── manifest.py         # Manifesto JSON de reprodutibilidade
│   └── pipeline.py             # Orquestração de alto nível (run_study)
└── tests/                      # Testes unitários do protocolo e contratos
```

## Registrando Modelos no Notebook (PyTorch / TensorFlow / Keras / Scikit-Learn)

O motor experimental conversa com os modelos exclusivamente via arrays NumPy e o contrato `GraphContext`:

```python
from media_gnn_study import register_model, ModelOutput

def meu_modelo_adapter(x_train, y_train, x_eval, graph_context, seed):
    # graph_context entrega p_train, p_eval, edge_index_train, edge_index_eval em NumPy
    ...
    return ModelOutput(
        probabilities=prob_eval,
        complexity=num_params,
        complexity_type="parameters",
        train_seconds=train_s,
        predict_seconds=pred_s,
    )

register_model(
    "meu_gnn",
    meu_modelo_adapter,
    display_name="Meu GNN Customizado",
    family="graph",
    role="relational",
    seeds=(42, 43, 44),
)
```

## Testes Locais (via `uv`)

```bash
uv run pytest
```
