---
title: Toxicity Target Type Identification
summary: O Classificador do Tipo de Alvo de Comentários Tóxicos Direcionados é um modelo que classifica o tipo (individual, grupo ou outro) de um determinado texto alvo.
---

O Classificador do Tipo de Alvo de Comentários Tóxicos Direcionados é um modelo que classifica o tipo (individual, grupo ou outro) de um determinado texto alvo.

Este modelo BERT é uma versão ajustada do [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased) no conjunto de dados [OLID-BR](https://huggingface.co/datasets/dougtrajano/olid-br).

## Visão geral

**Entrada:** Texto em português do Brasil

**Saída:** classificação multiclasse (individual, em grupo ou outra)

## Uso

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("dougtrajano/toxicity-target-type-identification")

model = AutoModelForSequenceClassification.from_pretrained("dougtrajano/toxicity-target-type-identification")
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

**Accuracy:** 0,7505

**Precision:** 0,7812

**Recall:** 0,7505

**F1-Score:** 0,7603

| Class | Precision | Recall | F1-Score | Support |
| :---: | :-------: | :----: | :------: | :-----: |
| `INDIVIDUAL` | 0,8850 | 0,7964 | 0,8384 | 609 |
| `GROUP` | 0,6766 | 0,6385 | 0,6570 | 213 |
| `OTHER` | 0,4518 | 0,7177 | 0,5545 | 124 |

## Procedimento de treinamento

### Hiperparâmetros de treinamento

Os seguintes hiperparâmetros foram usados durante o treinamento:

- learning_rate: 3.952388499692274e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 1993
- optimizer: Adam with betas=(0.9944095815441554,0.8750000522553327) and epsilon=1.8526084265228802e-07
- lr_scheduler_type: linear
- num_epochs: 30

### Versões dos frameworks

- Transformers 4.26.1
- Pytorch 1.10.2+cu113
- Datasets 2.9.0
- Tokenizers 0.13.2

## Sugestões e feedback

Se você tiver algum feedback sobre este modelo, crie um [issue](https://github.com/DougTrajano/ToChiquinho/issues/new) no GitHub.
