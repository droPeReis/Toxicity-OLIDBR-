---
title: Toxic Spans Detection
summary: Toxic Spans Detection is a Span Classification model that detects toxic spans in a given toxic text.
---

Toxic Spans Detection is a model that detects toxic spans in a given toxic text.

This BERT model is a fine-tuned version of [neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased) on the [OLID-BR dataset](https://huggingface.co/datasets/dougtrajano/olid-br).

## Overview

**Input:** Text in Brazilian Portuguese

**Output:** A list of integers representing the position of the toxic spans in the text

## Usage

Pending

## Limitations and bias

The following factors may degrade the model’s performance.

**Text Language**:  The model was trained on Brazilian Portuguese texts, so it may not work well with Portuguese dialects.

**Text Origin**: The model was trained on texts from social media and a few texts from other sources, so it may not work well on other types of texts.

## Trade-offs

Sometimes models exhibit performance issues under particular circumstances. In this section, we'll discuss situations in which you might discover that the model performs less than optimally, and should plan accordingly.

**Text Length**: The model was fine-tuned on texts with a word count between 1 and 178 words (average of 18 words). It may give poor results on texts with a word count outside this range.

## Performance

The model was evaluated on the test set of the [OLID-BR](https://dougtrajano.github.io/olid-br/) dataset.

**Precision:** 0.6876

**Recall:** 0.4918

**F1-Score:** 0.5734

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:

- learning_rate: 0.00038798590315954165
- dropout_rate: 0.3
- seed: 1993
- optimizer: Adam with betas=(0.9978242993498763,0.9988901284249041) and epsilon=3.12576102525027e-08
- lr_scheduler_type: linear
- num_epochs: 30
- weight_decay: 0.1

### Framework versions

- SpaCy 3.4.1
- SpaCy pt_core_news_lg model 3.4.0
- Datasets 2.9.0

## Provide Feedback

If you have any feedback on this model, please [open an issue](https://github.com/DougTrajano/ToChiquinho/issues/new) on GitHub.
