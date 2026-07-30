---
title: Toxic Comment Classification
summary: O Classificador de Comentários Tóxicos é um modelo que detecta se o texto é tóxico ou não.
---

O Classificador de Comentários Tóxicos é um modelo que detecta se o texto é tóxico ou não.

Este modelo BERT é uma versão ajustada do [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased) no conjunto de dados [OLID-BR](https://huggingface.co/datasets/dougtrajano/olid-br).

## Visão geral

**Entrada:** Texto em português do Brasil

**Saída:** Classificação binária (tóxica ou não tóxica)

## Uso

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("dougtrajano/toxic-comment-classification")

model = AutoModelForSequenceClassification.from_pretrained("dougtrajano/toxic-comment-classification")
```

## Limitações e viéses

Os seguintes fatores podem degradar o desempenho do modelo.

**Idioma do texto**: o modelo foi treinado em textos do português brasileiro, portanto, pode não funcionar bem com dialetos do português.

**Origem do texto**: o modelo foi treinado em textos de mídias sociais e alguns textos de outras fontes, portanto, pode não funcionar bem em outros tipos de texto.

## Trade-offs

Às vezes, os modelos exibem problemas de desempenho em circunstâncias específicas. Nesta seção, discutiremos situações nas quais você pode descobrir que o modelo tem desempenho inferior ao ideal e deve planejar de acordo.

**Tamanho do Texto**: O modelo foi ajustado em textos com contagem de palavras entre 1 e 178 palavras (média de 18 palavras). Pode dar resultados ruins em textos com uma contagem de palavras fora desse intervalo.

## Desempenho

O modelo foi avaliado no conjunto de teste do conjunto de dados [OLID-BR](https://dougtrajano.github.io/olid-br/).

**Accuracy:** 0,8578

**Precision:** 0,8594

**Recall:** 0,8578

**F1-Score:** 0,8580

| Class | Precision | Recall | F1-Score | Support |
| :---: | :-------: | :----: | :------: | :-----: |
| `NOT-OFFENSIVE` | 0,8886 | 0,8490 | 0,8683 | 1.775 |
| `OFFENSIVE` | 0,8233 | 0,8686 | 0,8453 | 1.438 |

## Procedimento de treinamento

### Hiperparâmetros de treinamento

Os seguintes hiperparâmetros foram usados durante o treinamento:

- learning_rate: 3.255788747459486e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 1993
- optimizer: Adam with betas=(0.8445637934160373,0.8338816842140165) and epsilon=2.527092625455385e-08
- lr_scheduler_type: linear
- num_epochs: 30
- label_smoothing_factor: 0.07158711257743958

### Versões dos frameworks

- Transformers 4.26.0
- Pytorch 1.10.2+cu113
- Datasets 2.9.0
- Tokenizers 0.13.2

## Sugestões e feedback

Se você tiver algum feedback sobre este modelo, crie um [issue](https://github.com/DougTrajano/ToChiquinho/issues/new) no GitHub.
