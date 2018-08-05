# InferSent

A Tensorflow implementation of Alexis Conneau's **[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364)** from EMNLP 2017.

# Desctription

*InferSent* is a *sentence embeddings* method that provides semantic representations for English sentences. It is trained on natural language inference data and generalizes well to many different tasks.

# To-do List

1. Save sentence embeddings permanently.

# Dataset

The dataset used for this task is [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/). Pretrained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) obtained from common crawl with 840B tokens used for words.

# Requirements

- Python>=3
- NumPy
- TensorFlow>=1.8

# Usage

Download dataset from [Stanford Natural Language Inference](https://nlp.stanford.edu/projects/snli/), then move `snli_1.0_train.jsonl`, `snli_1.0_dev.jsonl`, `snli_1.0_test.jsonl` into `./SNLI/raw data`.

```com
# move dataset to the right place
mkdir -p ./SNLI/raw\ data
mv snli_1.0_*.jsonl ./SNLI/raw\ data
```

Data preprocessing for convert source data into an easy-to-use format.

```python
python3 Utils.py
```

Default hyper-parameters have been stored in config file in the path of `./config/config.yaml`.

Training model:

```python
python3 Train.py
```