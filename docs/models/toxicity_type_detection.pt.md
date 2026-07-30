---
title: Toxicity Type Detection
summary: O Classificador dos Tipos de Linguagem Tóxica é um modelo que classifica rótulos de toxicidades predeterminados em um texto tóxico.
---

O Classificador dos Tipos de Linguagem Tóxica é um modelo que classifica rótulos de toxicidades predeterminados em um texto tóxico.

Rótulos de toxicidade: `health`, `ideology`, `insult`, `lgbtqphobia`, `other_lifestyle`, `physical_aspects`, `profanity_obscene`, `racism`, `sexism`, `xenophobia`

Este modelo BERT é uma versão ajustada do [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased) no conjunto de dados [OLID-BR](https://huggingface.co/datasets/dougtrajano/olid-br).

## Visão geral

**Entrada:** Texto em português do Brasil

**Saída:** Classificação multirrótulo (rótulos de toxicidade)

## Uso

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("dougtrajano/toxicity-type-detection")

model = AutoModelForSequenceClassification.from_pretrained("dougtrajano/toxicity-type-detection")
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

**Accuracy:** 0,4214

**Precision:** 0,8180

**Recall:** 0,7230

**F1-Score:** 0,7645

| Label | Precision | Recall | F1-Score | Support |
| :---: | :-------: | :----: | :------: | :-----: |
| `health` | 0,3182 | 0,1795 | 0,2295 | 39 |
| `ideology` | 0,6820 | 0,6842 | 0,6831 | 304 |
| `insult` | 0,9689 | 0,8068 | 0,8805 | 1351 |
| `lgbtqphobia` | 0,8182 | 0,5870 | 0,6835 | 92 |
| `other_lifestyle` | 0,4242 | 0,4118 | 0,4179 | 34 |
| `physical_aspects` | 0,4324 | 0,5783 | 0,4948 | 83 |
| `profanity_obscene` | 0,7482 | 0,7509 | 0,7496 | 562 |
| `racism` | 0,4737 | 0,3913 | 0,4286 | 23 |
| `sexism` | 0,5132 | 0,3391 | 0,4084 | 115 |
| `xenophobia` | 0,3333 | 0,4375 | 0,3784 | 32 |

## Procedimento de treinamento

### Hiperparâmetros de treinamento

Os seguintes hiperparâmetros foram usados durante o treinamento:

- learning_rate: 7.044186985160909e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 1993
- optimizer: Adam with betas=(0.9339215524915885,0.9916979096990963) and epsilon=3.4435900142455904e-07
- lr_scheduler_type: linear
- num_epochs: 30

### Versões dos frameworks

- Transformers 4.26.0
- Pytorch 1.10.2+cu113
- Datasets 2.9.0
- Tokenizers 0.13.2

## Sugestões e feedback

Se você tiver algum feedback sobre este modelo, crie um [issue](https://github.com/DougTrajano/ToChiquinho/issues/new) no GitHub.
