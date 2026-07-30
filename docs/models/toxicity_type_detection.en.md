---
title: Toxicity Type Detection
summary: This model detects the type(s) of toxicity(s) in a given text.
---

Toxicity Type Detection is a model that predicts the type(s) of toxicity(s) in a given text.

Toxicity Labels: `health`, `ideology`, `insult`, `lgbtqphobia`, `other_lifestyle`, `physical_aspects`, `profanity_obscene`, `racism`, `sexism`, `xenophobia`

This BERT model is a fine-tuned version of [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased) on the [OLID-BR dataset](https://huggingface.co/datasets/dougtrajano/olid-br).

## Overview

**Input:** Text in Brazilian Portuguese

**Output:** Multilabel classification (toxicity types)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("dougtrajano/toxicity-type-detection")

model = AutoModelForSequenceClassification.from_pretrained("dougtrajano/toxicity-type-detection")
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

**Accuracy:** 0.4214

**Precision:** 0.8180

**Recall:** 0.7230

**F1-Score:** 0.7645

| Label | Precision | Recall | F1-Score | Support |
| :---: | :-------: | :----: | :------: | :-----: |
| `health` | 0.3182 | 0.1795 | 0.2295 | 39 |
| `ideology` | 0.6820 | 0.6842 | 0.6831 | 304 |
| `insult` | 0.9689 | 0.8068 | 0.8805 | 1351 |
| `lgbtqphobia` | 0.8182 | 0.5870 | 0.6835 | 92 |
| `other_lifestyle` | 0.4242 | 0.4118 | 0.4179 | 34 |
| `physical_aspects` | 0.4324 | 0.5783 | 0.4948 | 83 |
| `profanity_obscene` | 0.7482 | 0.7509 | 0.7496 | 562 |
| `racism` | 0.4737 | 0.3913 | 0.4286 | 23 |
| `sexism` | 0.5132 | 0.3391 | 0.4084 | 115 |
| `xenophobia` | 0.3333 | 0.4375 | 0.3784 | 32 |

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:

- learning_rate: 7.044186985160909e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 1993
- optimizer: Adam with betas=(0.9339215524915885,0.9916979096990963) and epsilon=3.4435900142455904e-07
- lr_scheduler_type: linear
- num_epochs: 30

### Framework versions

- Transformers 4.26.0
- Pytorch 1.10.2+cu113
- Datasets 2.9.0
- Tokenizers 0.13.2

## Provide Feedback

If you have any feedback on this model, please [open an issue](https://github.com/DougTrajano/ToChiquinho/issues/new) on GitHub.
