---
title: Toxicity Target Type Identification
summary: This model classifies the type (individual, group, or other) of a given targeted text.
---

Toxicity Target Type Identification is a model that classifies the type (individual, group, or other) of a given targeted text.

This BERT model is a fine-tuned version of [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased) on the [OLID-BR dataset](https://huggingface.co/datasets/dougtrajano/olid-br).

## Overview

**Input:** Text in Brazilian Portuguese

**Output:** Multiclass classification (individual, group, or other)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("dougtrajano/toxicity-target-type-identification")

model = AutoModelForSequenceClassification.from_pretrained("dougtrajano/toxicity-target-type-identification")
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

**Accuracy:** 0.7505

**Precision:** 0.7812

**Recall:** 0.7505

**F1-Score:** 0.7603

| Class | Precision | Recall | F1-Score | Support |
| :---: | :-------: | :----: | :------: | :-----: |
| `INDIVIDUAL` | 0.8850 | 0.7964 | 0.8384 | 609 |
| `GROUP` | 0.6766 | 0.6385 | 0.6570 | 213 |
| `OTHER` | 0.4518 | 0.7177 | 0.5545 | 124 |

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:

- learning_rate: 3.952388499692274e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 1993
- optimizer: Adam with betas=(0.9944095815441554,0.8750000522553327) and epsilon=1.8526084265228802e-07
- lr_scheduler_type: linear
- num_epochs: 30

### Framework versions

- Transformers 4.26.1
- Pytorch 1.10.2+cu113
- Datasets 2.9.0
- Tokenizers 0.13.2

## Provide Feedback

If you have any feedback on this model, please [open an issue](https://github.com/DougTrajano/ToChiquinho/issues/new) on GitHub.
