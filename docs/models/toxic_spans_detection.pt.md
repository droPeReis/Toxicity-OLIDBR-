---
title: Toxic Spans Detection
summary: O Detector das partes tóxicas do texto é um modelo que detecta spans tóxicos em um determinado texto tóxico.
---

O Detector das partes tóxicas do texto é um modelo que detecta spans tóxicos em um determinado texto tóxico.

Este modelo BERT é uma versão ajustada do [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased) no conjunto de dados [OLID-BR](https://huggingface.co/datasets/dougtrajano/olid-br).

## Visão geral

**Entrada:** Texto em português do Brasil

**Saída:** Uma lista com os indices de cada span tóxico no texto

## Uso

Pendente

## Limitações e viéses

Os seguintes fatores podem degradar o desempenho do modelo.

**Idioma do texto**: o modelo foi treinado em textos do português brasileiro, portanto, pode não funcionar bem com dialetos do português.

**Origem do texto**: o modelo foi treinado em textos de mídias sociais e alguns textos de outras fontes, portanto, pode não funcionar bem em outros tipos de texto.

## Trade-offs

Às vezes, os modelos exibem problemas de desempenho em circunstâncias específicas. Nesta seção, discutiremos situações nas quais você pode descobrir que o modelo tem desempenho inferior ao ideal e deve planejar de acordo.

**Tamanho do Texto**: O modelo foi ajustado em textos com contagem de palavras entre 1 e 178 palavras (média de 18 palavras). Pode dar resultados ruins em textos com uma contagem de palavras fora desse intervalo.

## Desempenho

O modelo foi avaliado no conjunto de teste do conjunto de dados [OLID-BR](https://dougtrajano.github.io/olid-br/).

**Precision:** 0,6876

**Recall:** 0,4918

**F1-Score:** 0,5734

## Procedimento de treinamento

### Hiperparâmetros de treinamento

Os seguintes hiperparâmetros foram usados durante o treinamento:

- learning_rate: 0.00038798590315954165
- dropout_rate: 0.3
- seed: 1993
- optimizer: Adam with betas=(0.9978242993498763,0.9988901284249041) and epsilon=3.12576102525027e-08
- lr_scheduler_type: linear
- num_epochs: 30
- weight_decay: 0.1

### Versões dos frameworks

- SpaCy 3.4.1
- SpaCy pt_core_news_lg model 3.4.0
- Datasets 2.9.0

## Sugestões e feedback

Se você tiver algum feedback sobre este modelo, crie um [issue](https://github.com/DougTrajano/ToChiquinho/issues/new) no GitHub.
