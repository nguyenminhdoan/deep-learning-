# COMP 263 Term Project: NLP with Disaster Tweets

Binary classification of tweets: does a tweet describe a real disaster (1) or not (0)?

**Group members:** _TODO: add names and student numbers_

## Setup

```
pip install -r requirements.txt
```

Tested with Python 3.11, TensorFlow 2.19 (Keras 3), gensim 4.4.

## Dataset

Kaggle competition [NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/data).
Download `train.csv` and `test.csv` and place them in `data/`.

- `train.csv`: 7,613 labelled tweets. Split once into train / validation / test
  (80 / 10 / 10, stratified, seed 42). The assignment is stored in
  `data/splits.csv` (created on first run) so **every experiment uses the same
  held-out tweets**.
- `test.csv`: 3,263 tweets without labels. Used only as extra unlabelled text
  for Word2Vec.

## Project layout

```
common/                 shared by all experiments
  config.py             paths, seed, split ratios, hyperparameters
  data.py               TweetCleaner, DisasterTweetDataset (shared split), Vocabulary
  evaluation.py         metrics, confusion matrix, training-curve plots
eda.py                  exploratory data analysis
exp2_unsupervised/      Experiment 2: Word2Vec feature extraction
  word2vec_trainer.py   Word2VecTrainer (training, embedding matrix, t-SNE)
  classifier.py         BiLSTMClassifier (downstream supervised model)
  run_experiment.py     runs the whole experiment
results/                all generated output
```

## Running

Run everything from the project root.

```
python eda.py                                        # -> results/eda/
python -m exp2_unsupervised.run_experiment           # -> results/exp2_unsupervised/  (~30-60 min on CPU)
python -m exp2_unsupervised.run_experiment --quick   # 1 seed, a few minutes
```

## Experiment 2: unsupervised feature extraction with Word2Vec

1. **Unsupervised pre-training.** Word2Vec (skip-gram and CBOW, 100-d) is
   trained on tweet text only: the training split plus Kaggle's unlabelled
   `test.csv`. No labels are used, and the validation / test splits are excluded.
2. **Inspection.** Nearest neighbours and a t-SNE map of the learned vectors.
3. **Linear probe.** Logistic regression on the mean word vector of each tweet.
4. **Transfer learning.** One fixed BiLSTM is trained with three kinds of
   embedding layer: random initialisation (baseline), frozen Word2Vec
   (feature extraction) and fine-tuned Word2Vec. Each is repeated over 3 seeds.
5. **Label efficiency.** The comparison is repeated with 10 / 25 / 50 % of the
   training labels.

Output in `results/exp2_unsupervised/`:

| File | Content |
|---|---|
| `summary.md`, `summary.csv` | accuracy / precision / recall / F1, mean ± std over seeds |
| `all_runs.csv` | every individual run |
| `f1_comparison.png` | test F1 per embedding type |
| `training_curves.png` | train / validation accuracy and loss |
| `confusion_*.png` | confusion matrix per variant |
| `label_efficiency.png`, `label_efficiency_runs.csv` | F1 vs. number of labelled tweets |
| `linear_probe.csv` | logistic regression on mean Word2Vec vectors |
| `nearest_neighbours.txt`, `tsne_*.png` | qualitative view of the embeddings |
| `word2vec_*.kv` | trained vectors (gensim KeyedVectors) |

## Using the shared pipeline in another experiment

```python
from common.data import DisasterTweetDataset
from common.evaluation import compute_metrics, plot_confusion_matrix, plot_training_curves

data = DisasterTweetDataset()
train, val, test = data.subset("train"), data.subset("val"), data.subset("test")
# columns: id, keyword, location, text, target, tokens, split
metrics = compute_metrics(test["target"], predicted_probabilities)
```
