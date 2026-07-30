---
title: Toxicity Target Classification
summary: This model classifies if a given text is targeted or not.
---

Toxicity Target Classification is a model that classifies if a given text is targeted or not.

This BERT model is a fine-tuned version of [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased) on the [OLID-BR dataset](https://huggingface.co/datasets/dougtrajano/olid-br).

## Overview

**Input:** Text in Brazilian Portuguese

**Output:** Binary classification (targeted or untargeted)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("dougtrajano/toxicity-target-classification")

model = AutoModelForSequenceClassification.from_pretrained("dougtrajano/toxicity-target-classification")
```

## Limitations and bias

The following factors may degrade the model’s performance.

**Text Language**:  The model was trained on Brazilian Portuguese texts, so it may not work well with Portuguese dialects.

**Text Origin**: The model was trained on texts from social media and a few texts from other sources, so it may not work well on other types of texts.

## Trade-offs

Sometimes models exhibit performance issues under particular circumstances. In this section, we'll discuss situations in which you might discover that the model performs less than optimally, and should plan accordingly.

**Text Length**: The model was fine-tuned on texts with a word count between 1 and 178 words (average of 18 words). It may give poor results on texts with a word count outside this range.

## Performance

The model was evaluated on the test set of the [OLID-BR](https://dougtrajano.github.io/olid-br/) dataset.

**Accuracy:** 0.6864

**Precision:** 0.6882

**Recall:** 0.6864

**F1-Score:** 0.6872

| Class | Precision | Recall | F1-Score | Support |
| :---: | :-------: | :----: | :------: | :-----: |
| `UNTARGETED` | 0.4912 | 0.5011 | 0.4961 | 443 |
| `TARGETED INSULT` | 0.7759 | 0.7688 | 0.7723 | 995 |

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:

- learning_rate: 4.174021560583183e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 1993
- optimizer: Adam with betas=(0.9360294728287728,0.9974781444436187) and epsilon=8.016624612627008e-07
- lr_scheduler_type: linear
- num_epochs: 30
- label_smoothing_factor: 0.09936835309930625

### Framework versions

- Transformers 4.26.0
- Pytorch 1.10.2+cu113
- Datasets 2.9.0
- Tokenizers 0.13.2

## Provide Feedback

If you have any feedback on this model, please [open an issue](https://github.com/DougTrajano/ToChiquinho/issues/new) on GitHub.
