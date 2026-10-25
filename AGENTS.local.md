# AGENTS.local.md

Diretrizes locais do repositório `veritas-ml` (framework experimental auditável e agnóstico a modelos para benchmark de representações e estruturas relacionais).

## Stack e Ambiente
- **Linguagem:** Python >= 3.11 (ambiente local gerenciado via `uv`).
- **Gerenciador de Pacotes/Testes:** `uv run pytest`.
- **Arquitetura Modular (`src/veritas_ml/`):**
  - `core/`: Contratos (`DatasetContext`, `FeatureContext`, `GraphContext`, `ExperimentPlan`, `ExperimentResult`, `StudyResult`), configurações agnósticas de protocolo e álgebra de grafos pura em NumPy/SciPy.
  - `data/`: Leitura validada por SHA-256 e geração de folds agrupados por franquia (`grupo_franquia`), com suporte a fallback determinístico local (`allow_local_fallback=True`) quando `particoes_franquias.csv` não está presente.
  - `features/`: Extratores lexicais (TF-IDF/SVD/NMF), linguísticos (spaCy) e encoders neurais cacheados.
  - `models/`: Registro 100% agnóstico (`register_model`, `register_special_experiment`, `run_model`) e contratos `ModelOutput`/`ModelSpec`. O framework **não contém implementações de modelos**; todas as arquiteturas (PyTorch, PyG, Scikit-Learn/XGBoost, TensorFlow/Keras) e hiperparâmetros vivem no notebook experimental (`notebooks/01_estudo_local.ipynb`).
  - `engine/`: Catálogo fatorial de bundles, montagem dinâmica do `ExperimentPlan` a partir do registry, sanity checks e runner de experimentos.
  - `evaluation/`: Diagnósticos relacionais (homofilia, curvas de aprendizado), agregação de `StudyResult`, renderização e manifesto de reprodutibilidade.

## Aprendizados Acumulados
- **Separação Total entre Framework e Modelos:** Nenhum módulo de `src/veritas_ml/` define arquiteturas de modelos, hiperparâmetros de treino (`EPOCHS`, `LR`, `HIDDEN`, `DROPOUT`) ou nomes de modelos fixos em `config.py`. Os modelos são definidos no notebook e injetados via `register_model(...)` e `register_special_experiment(...)`.
- **Bug corrigido em `assemble_study_result` (`evaluation/study.py`):** Na extração original do notebook monolítico, o bloco condicional de `graph_operator_comparison` dentro de `assemble_study_result` tentava chamar `specs.append(...)` (variável inexistente naquele escopo, pertencente a `build_plot_specs`) em vez de `table_specs.append(...)`, o que causava `NameError` quando diagnósticos de operadores homogêneos estavam presentes. Corrigido e coberto em `tests/test_evaluation.py`.
