# Migração do notebook monolítico

Este corte foi extraído do notebook central v5. A intenção é preservar comportamento primeiro e refatorar internamente depois.

- O notebook deixa de ser a fonte autoritativa de funções/classes.
- `src/media_gnn_study` passa a ser a fonte autoritativa.
- O notebook declara ambiente e chama a pipeline.
- Dataset e resultados continuam fora do Git.
- Mudanças metodológicas devem entrar como novos experimentos/configurações, com testes quando afetarem protocolo indutivo ou fingerprints.
