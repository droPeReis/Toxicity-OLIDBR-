---
title: Toxicity Target Classification
summary: O Classificador de Comentários Direcionados é um modelo que classifica se um determinado texto é alvo ou não.
---

O Classificador de Comentários Tóxicos Direcionados é um modelo que classifica se um determinado texto é alvo ou não.

Este modelo BERT é uma versão ajustada do [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased) no conjunto de dados [OLID-BR](https://huggingface.co/datasets/dougtrajano/olid-br).

## Visão geral

**Entrada:** Texto em português do Brasil

**Saída:** classificação binária (direcionada ou não direcionada)

## Uso

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("dougtrajano/toxicity-target-classification")

model = AutoModelForSequenceClassification.from_pretrained("dougtrajano/toxicity-target-classification")
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

**Precisão:** 0,6864

**Precisão:** 0,6882

**Lembrança:** 0,6864

**F1-Pontuação:** 0,6872

| Class | Precision | Recall | F1-Score | Support |
| :---: | :-------: | :----: | :------: | :-----: |
| `UNTARGETED` | 0,4912 | 0,5011 | 0,4961 | 443 |
| `TARGETED INSULT` | 0,7759 | 0,7688 | 0,7723 | 995 |

## Procedimento de treinamento

### Hiperparâmetros de treinamento

Os seguintes hiperparâmetros foram usados durante o treinamento:

- learning_rate: 4.174021560583183e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 1993
- optimizer: Adam with betas=(0.9360294728287728,0.9974781444436187) and epsilon=8.016624612627008e-07
- lr_scheduler_type: linear
- num_epochs: 30
- label_smoothing_factor: 0.09936835309930625

### Versões dos frameworks

- Transformers 4.26.0
- Pytorch 1.10.2+cu113
- Datasets 2.9.0
- Tokenizers 0.13.2

## Sugestões e feedback

Se você tiver algum feedback sobre este modelo, crie um [issue](https://github.com/DougTrajano/ToChiquinho/issues/new) no GitHub.
