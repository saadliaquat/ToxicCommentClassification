# Toxic Comment Classification

Multi-label classification of Wikipedia comments into six toxicity categories, comparing a classical baseline, a sequence model, and a fine-tuned transformer.

**Labels:** `toxic` · `severe_toxic` · `obscene` · `threat` · `insult` · `identity_hate`

A comment can belong to any combination of these labels, so each model predicts six independent binary outputs.

## Dataset

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Kaggle)

- **Train:** 159,571 labelled comments
- **Test:** 63,978 comments (after removing rows labelled `-1`, which have no ground truth)
- The labels are heavily imbalanced: `toxic` is common, while `threat` and `severe_toxic` are rare.

Download `train.csv`, `test.csv` and `test_labels.csv` into a `data/` folder.

## Models

### 1. Naive Bayes (baseline)
- Preprocessing: lowercasing, and removing punctuation, numbers and stopwords
- Features: `CountVectorizer` and `TfidfVectorizer`, compared
- Classifier: `MultinomialNB` wrapped in `MultiOutputClassifier` (one model per label)

### 2. LSTM
- Preprocessing: the same as above, plus URL removal and filtering out very short comments
- Tokenized and padded to 100 tokens with a 10k-word vocabulary
- Architecture: Embedding → LSTM (50 units) → Dense (6, sigmoid)
- Trained with binary cross-entropy and Adam

### 3. BERT
- Fine-tuned `bert-base-uncased` with a 6-label classification head
- Light preprocessing (lowercasing and whitespace cleanup); the BERT tokenizer handles the rest
- AdamW (lr 2.5e-5) with a linear learning-rate decay, trained for 1 epoch
- Outputs pass through a sigmoid with a 0.5 threshold per label

## Results

| Model | Split | Micro-F1 | Macro-F1 |
|---|---|---|---|
| Naive Bayes (Count) | Test | 0.53 | 0.35 |
| Naive Bayes (TF-IDF) | Test | 0.21 | 0.10 |
| LSTM | Validation* | 0.71 | 0.48 |
| **BERT (fine-tuned)** | **Test** | **0.66** | **0.48** |

\*The LSTM was evaluated on its own validation split, so its scores are not directly comparable to the test scores.

## Key Findings

- **Accuracy is misleading here.** With imbalanced multi-label data, a model can score high accuracy by rarely predicting toxicity. TF-IDF Naive Bayes reached about 90% accuracy but only 0.21 micro-F1, so we compare models on per-class precision, recall and F1.
- **Count vs TF-IDF:** count features produced more balanced predictions. TF-IDF was very conservative: high precision, very low recall.
- **BERT performed best overall,** improving micro-F1 by about 25% over the Naive Bayes baseline, thanks to its bidirectional understanding of context.
- **Rare classes remain hard.** BERT scored 0 F1 on `severe_toxic`. On `identity_hate` it had high precision (0.85) but low recall (0.29).

## Future Work

- Handle class imbalance with class-weighted loss or a tuned threshold per label
- Train BERT for more epochs (limited to 1 here by Colab GPU time limits)
- Try lighter transformers such as DistilBERT, or newer models such as RoBERTa or DeBERTa

## Project Structure

```
├── 01_naive_bayes.ipynb
├── 02_lstm.ipynb
├── 03_bert.ipynb
└── data/              # not included, download from Kaggle
```


- Saad Liaquat
- [Teammate name]
- [Teammate name]
