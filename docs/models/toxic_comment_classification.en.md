---
title: Toxic Comment Classification
summary: Toxic Comment Classification is a model that detects if the text is toxic or not.
---

Toxic Comment Classification is a model that detects if the text is toxic or not.

This BERT model is a fine-tuned version of [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased) on the [OLID-BR dataset](https://huggingface.co/datasets/dougtrajano/olid-br).

## Overview

**Input:** Text in Brazilian Portuguese

**Output:** Binary classification (toxic or not toxic)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("dougtrajano/toxic-comment-classification")

model = AutoModelForSequenceClassification.from_pretrained("dougtrajano/toxic-comment-classification")
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

**Accuracy:** 0.8578

**Precision:** 0.8594

**Recall:** 0.8578

**F1-Score:** 0.8580

| Class | Precision | Recall | F1-Score | Support |
| :---: | :-------: | :----: | :------: | :-----: |
| `NOT-OFFENSIVE` | 0.8886 | 0.8490 | 0.8683 | 1,775 |
| `OFFENSIVE` | 0.8233 | 0.8686 | 0.8453 | 1,438 |

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:

- learning_rate: 3.255788747459486e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 1993
- optimizer: Adam with betas=(0.8445637934160373,0.8338816842140165) and epsilon=2.527092625455385e-08
- lr_scheduler_type: linear
- num_epochs: 30
- label_smoothing_factor: 0.07158711257743958

### Framework versions

- Transformers 4.26.0
- Pytorch 1.10.2+cu113
- Datasets 2.9.0
- Tokenizers 0.13.2

## Provide Feedback

If you have any feedback on this model, please [open an issue](https://github.com/DougTrajano/ToChiquinho/issues/new) on GitHub.
