# 🛡️ SpamShield — AI Spam Email Classifier

A full-stack spam detection system powered by an ensemble of three ML models: **Naive Bayes**, **Support Vector Machine (LinearSVC)**, and **Logistic Regression** — all trained on TF-IDF bi-gram features.

---

## Features

- **Ensemble classification** — weighted vote across 3 independent models for higher accuracy
- **Real-time inference** — sub-50ms predictions via Flask REST API
- **Text preprocessing** — URL normalisation, stopword removal, Porter stemming
- **Confidence scoring** — probability output shown per model and combined
- **Spam signal extraction** — highlights the specific words that triggered spam detection
- **One-click retraining** — retrain all models in-browser via the UI
- **~97% accuracy** on the UCI SMS Spam Collection dataset

---

## Tech Stack

| Layer | Tech |
|-------|------|
| **ML Models** | Naive Bayes, LinearSVC, Logistic Regression (scikit-learn) |
| **Features** | TF-IDF (1–2 grams, 20k features, sublinear TF) |
| **Backend** | Python 3 · Flask · Flask-CORS |
| **Preprocessing** | NLTK (stopwords, Porter stemmer) |
| **Frontend** | Vanilla HTML/CSS/JS — dark glassmorphism UI |

---

## Project Structure

```
spam-mail-checker/
├── app.py                  # Flask API server
├── requirements.txt
├── model/
│   ├── train_model.py      # Training script (Naive Bayes, SVM, LR)
│   ├── spam_data.csv       # Dataset (auto-downloaded on first run)
│   ├── nb_model.pkl        # Trained Naive Bayes pipeline
│   ├── svm_model.pkl       # Trained SVM pipeline
│   └── lr_model.pkl        # Trained Logistic Regression pipeline
├── templates/
│   └── index.html          # Main UI template
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## Setup & Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the models (first time only; auto-downloads the dataset)
python model/train_model.py

# 3. Start the server
python app.py
```

Then open **http://localhost:5000** in your browser.

> The server will auto-train on first launch if no `.pkl` files are found.

---

## API Reference

### `POST /api/check`
Classify an email.

**Body:** `{ "text": "your email content here" }`

**Response:**
```json
{
  "is_spam": true,
  "confidence": 94.3,
  "label": "SPAM",
  "model_votes": {
    "naive_bayes": { "label": "spam", "confidence": 91.2 },
    "svm":         { "label": "spam", "confidence": 97.1 },
    "logistic_regression": { "label": "spam", "confidence": 94.8 }
  },
  "spam_signals": ["free", "winner", "claim", "prize"],
  "latency_ms": 12.4
}
```

### `POST /api/train`
Retrain all models.

### `GET /api/status`
Check if models are loaded.

### `GET /api/examples`
Get curated spam/ham example texts.

---

## How the Ensemble Works

1. **TF-IDF vectorisation** — each model has its own TF-IDF step (bi-grams, 20k features)
2. **Calibrated probabilities** — SVM uses `CalibratedClassifierCV` so all models output true probabilities
3. **Weighted vote** — `NB × 0.25 + SVM × 0.45 + LR × 0.30`
4. **Threshold** — if weighted probability ≥ 0.5, classified as spam
