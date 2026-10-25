# AGENTS.local.md

Diretrizes locais do repositório `media-gnn-study` (classificação multirrótulo de obras de mídia e benchmark relacional em grafos).

## Stack e Ambiente
- **Linguagem:** Python >= 3.11 (ambiente local gerenciado via `uv`).
- **Gerenciador de Pacotes/Testes:** `uv run pytest`.
- **Arquitetura Modular (`src/media_gnn_study/`):**
  - `core/`: Contratos (`DatasetContext`, `FeatureContext`, `GraphContext`, `ExperimentPlan`, `ExperimentResult`, `StudyResult`), configurações agnósticas e álgebra de grafos pura em NumPy/SciPy.
  - `data/`: Leitura validada por SHA-256 e geração de folds agrupados por franquia (`grupo_franquia`).
  - `features/`: Extratores lexicais (TF-IDF/SVD/NMF), linguísticos (spaCy) e encoders neurais cacheados.
  - `models/`: Registro plugável (`register_model`, `run_model`) e contrato universal `ModelOutput`, permitindo injetar modelos de qualquer framework (PyTorch, TensorFlow/Keras, Scikit-Learn) a partir de notebooks.
  - `engine/`: Catálogo fatorial de bundles, sanity checks e runner de experimentos.
  - `evaluation/`: Diagnósticos relacionais (homofilia, curvas de aprendizado), agregação de `StudyResult`, renderização e manifesto de reprodutibilidade.

## Aprendizados Acumulados
- **Desacoplamento de Frameworks de Deep Learning:** Nenhum módulo de `core/`, `data/`, `engine/` ou `evaluation/` deve importar `torch` ou `tensorflow` no topo do arquivo. Toda comunicação entre o motor experimental e os modelos deve ocorrer exclusivamente via `np.ndarray`, matrizes esparsas SciPy e o contrato `GraphContext`.
- **Bug corrigido em `assemble_study_result` (`evaluation/study.py`):** Na extração original do notebook monolítico, o bloco condicional de `graph_operator_comparison` dentro de `assemble_study_result` tentava chamar `specs.append(...)` (variável inexistente naquele escopo, pertencente a `build_plot_specs`) em vez de `table_specs.append(...)`, o que causava `NameError` quando diagnósticos de operadores homogêneos estavam presentes. Corrigido e coberto em `tests/test_evaluation.py`.
