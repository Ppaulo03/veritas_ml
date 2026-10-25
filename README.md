# Media GNN Study

Código modular do estudo de classificação multilabel de obras de mídia. O dataset e os resultados permanecem no Google Drive; o repositório contém somente código, configuração experimental e testes.

## Objetivo

Pergunta central: **relacionar obras em um grafo ajuda a classificá-las melhor do que analisar cada sinopse isoladamente?**

Métricas primárias: micro-F1 e macro-F1. O conjunto de auditoria permanece reservado até o congelamento da metodologia.

## Estrutura

```text
src/media_gnn_study/
  config.py       # protocolo, caminhos e hiperparâmetros
  contracts.py    # DatasetContext / ExperimentPlan / Result contracts
  data.py         # dataset congelado e folds agrupados
  features.py     # MiniLM, TF-IDF/SVD, emoção, entidades, tópicos etc.
  models.py       # Linear, MLP, Fixed-kNN, GCN, GraphSAGE, GATv2, hetero
  plan.py         # catálogo de bundles e experimentos focais
  engine.py       # execução incremental + persistência
  diagnostics.py  # learning curve e diagnósticos relacionais
  study.py        # montagem do StudyResult
  render.py       # renderer agnóstico
  pipeline.py     # orquestração de alto nível
```

## Colab

1. Monte o Drive.
2. Clone este repositório.
3. Defina `GNN_PROJECT_ROOT=/content/drive/MyDrive/Pesquisa GNN Midias` **antes de importar o pacote**.
4. Instale em modo editável: `pip install -e ".[full]"`.
5. Execute `from media_gnn_study import run_study; result = run_study()`.

Veja `notebooks/00_estudo_central_colab.ipynb`.

## Execução incremental

Por padrão:

- `GNN_INCREMENTAL_REUSE=1`
- `GNN_INCREMENTAL_REUSE_STRICT=1`
- `GNN_INCREMENTAL_BASE_RUN_ID=20261025T142618Z_d9aa19`

O modo estrito falha se a base não puder ser validada; não cai silenciosamente para rerun completo. Para a rodada final congelada, use `GNN_INCREMENTAL_REUSE=0`.

## Testes

```bash
pytest -q
```

Os testes iniciais protegem as propriedades mais sensíveis do protocolo: normalização da propagação kNN, ausência de validação→treino/validação e poda top-k sem rótulos.

## Versionamento para evidência final

Na rodada final, registre o commit Git junto ao manifesto da execução e faça checkout desse commit no Colab. Evite executar a evidência final a partir de `main` mutável.
