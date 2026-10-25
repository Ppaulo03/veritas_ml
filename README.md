# Media GNN Study

Framework experimental agnóstico para classificação multirrótulo de obras de mídia e benchmark relacional em grafos. O dataset e os artefatos de execução permanecem no Google Drive; o repositório contém contratos, protocolo indutivo, extração de representações, motor de avaliação e testes.

## Objetivo

Pergunta central: **relacionar obras em um grafo ajuda a classificá-las melhor do que analisar cada sinopse isoladamente?**

Métricas primárias: micro-F1 e macro-F1. O conjunto de auditoria permanece reservado até o congelamento da metodologia.

## Estrutura Modular

```text
src/media_gnn_study/
├── core/
│   ├── config.py       # protocolo, caminhos e hiperparâmetros
│   ├── contracts.py    # DatasetContext / FeatureContext / GraphContext / ExperimentResult / StudyResult
│   └── graph.py        # álgebra de grafos indutiva pura em NumPy/SciPy (kNN, poda top-k, incidência bipartida)
├── data/
│   ├── loader.py       # leitura do CSV, limpeza e validação criptográfica SHA-256
│   └── splits.py       # folds agrupados por franquia sem vazamento
├── features/
│   ├── encoders.py     # MiniLM e XLM-RoBERTa emoção (com cache em disco)
│   ├── linguistic.py   # spaCy NER, pares verbo-objeto e métricas sintáticas
│   └── lexical.py      # TF-IDF, SVD, NMF e montagem fold-local do FeatureContext
├── models/
│   ├── base.py         # contrato ModelOutput e protocolo ModelAdapterFn
│   ├── metrics.py      # métricas multirrótulo e por classe
│   ├── registry.py     # catálogo plugável (register_model / run_model)
│   └── backends/       # implementações de referência (Classical e PyTorch/PyG)
├── engine/
│   ├── plan.py         # catálogo de 55 bundles e experimentos focais
│   ├── sanity.py       # verificações pré-execução do protocolo indutivo
│   └── runner.py       # execução incremental agnóstica a framework
├── evaluation/
│   ├── diagnostics.py  # homofilia × ganho, learning curves e subgrafos qualitativos
│   ├── study.py        # montagem do StudyResult e fronteira de Pareto
│   ├── render.py       # renderizador declarativo de tabelas e figuras
│   └── manifest.py     # manifesto JSON de reprodutibilidade
└── pipeline.py         # orquestração de alto nível (run_study)
```

## Injetando Modelos Customizados (TensorFlow / Keras / PyTorch) no Notebook

O motor experimental conversa com os modelos exclusivamente via arrays NumPy e o contrato `GraphContext`:

```python
from media_gnn_study import register_model, ModelOutput, run_study

def meu_modelo_tensorflow(x_train, y_train, x_eval, graph_context, seed):
    # graph_context entrega p_train, p_eval, edge_index_train, edge_index_eval em NumPy
    # Treine seu modelo em TensorFlow/Keras, PyTorch ou JAX:
    ...
    return ModelOutput(
        probabilities=prob_eval,
        complexity=num_params,
        complexity_type="parameters",
        train_seconds=train_s,
        predict_seconds=pred_s,
    )

register_model(
    "tf_gnn",
    meu_modelo_tensorflow,
    metadata={"display_name": "TensorFlow GNN", "family": "graph", "role": "relational"},
)
```

## Testes Locais (via `uv`)

```bash
uv run pytest
```
