# Veritas ML (`veritas-ml`)

Framework experimental auditável e 100% agnóstico a modelos para benchmark de representações, fusão fatorial de features e aprendizado relacional sem vazamento de dados.

- **Framework (`src/veritas_ml/`):** Contratos imutáveis, protocolo indutivo auditado por SHA-256 (zero vazamento validação→treino), álgebra opcional de grafos em NumPy/SciPy, extração de representações, motor de avaliação e renderizador declarativo.
- **Notebook (`notebooks/01_estudo_local.ipynb`):** Definição das arquiteturas de modelos (PyTorch, PyG, Scikit-Learn/XGBoost ou TensorFlow/Keras), hiperparâmetros de treino e registro no framework.

## Estrutura do Repositório

```text
├── notebooks/
│   └── 01_estudo_local.ipynb   # Arquiteturas de modelos, registro e execução local
├── src/veritas_ml/
│   ├── core/
│   │   ├── config.py           # Protocolo experimental e caminhos (sem modelos acoplados)
│   │   ├── contracts.py        # DatasetContext / FeatureContext / GraphContext / ExperimentResult / StudyResult
│   │   └── graph.py            # Álgebra indutiva de grafos pura em NumPy/SciPy (kNN, poda top-k, incidência bipartida)
│   ├── data/
│   │   ├── loader.py           # Leitura do CSV, limpeza e validação criptográfica SHA-256
│   │   └── splits.py           # Folds agrupados sem vazamento entre grupos/franquias
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

O motor experimental conversa com os modelos exclusivamente via arrays NumPy e o contrato opcional `GraphContext`:

```python
from veritas_ml import register_model, ModelOutput

def meu_modelo_adapter(x_train, y_train, x_eval, graph_context, seed):
    # Treine qualquer modelo (MLP, Árvores, Keras, GNN) recebendo arrays NumPy:
    ...
    return ModelOutput(
        probabilities=prob_eval,
        complexity=num_params,
        complexity_type="parameters",
        train_seconds=train_s,
        predict_seconds=pred_s,
    )

register_model(
    "meu_modelo",
    meu_modelo_adapter,
    display_name="Meu Modelo Customizado",
    family="neural",
    role="individual",
    seeds=(42, 43, 44),
)
```

## Testes Locais (via `uv`)

```bash
uv run pytest
```
