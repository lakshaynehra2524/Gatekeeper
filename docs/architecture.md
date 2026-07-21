# Architecture & ML Design

## 1. System position

```
┌───────────┐  PCM chunks  ┌────────────────────────── Android phone ─────────────────────────┐
│  wearable │ ───────────▶ │ ┌────────┐ transcript ┌──────────────────┐ FORWARD ┌───────────┐ │
│  device   │  (BLE/BT)    │ │ speech │ fragments  │  EDGE GATEKEEPER │────────▶│ deeper    │ │
└───────────┘              │ │ to-text│ ──────────▶│  (this project)  │         │intelligence│ │
                           │ └────────┘ incremental└──────────────────┘ REJECT  └───────────┘ │
                           │              (out of scope)     │           (dropped locally)    │
                           └─────────────────────────────────┼────────────────────────────────┘
                                                             ▼
                                            everything runs on-device, no network
```

Out of scope per the brief (suggestions in §5): wearable link, PCM transport,
the STT engine, and the downstream assistant itself.

## 2. Gatekeeper pipeline (per fragment)

```
 transcript fragment
        │
        ▼
 ┌─────────────────┐   pure filler ("haha", "cool cool")
 │ 1. normalise +  │ ─────────────────────────────────────▶ REJECT (no ML run)
 │  back-channel   │
 │  regex filter   │
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ 2. TF-IDF       │  word 1–2 grams + char_wb 3–5 grams
 │  vectorizer     │  (~7k dims, sparse)
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ 3. Logistic     │  6-class probabilities
 │  Regression     │  p_meaningful = 1 − p(ordinary)
 └────────┬────────┘
          ▼
 ┌─────────────────┐   cosine ≥ 0.45 vs any of the last 12
 │ 4. duplicate    │   forwarded moments
 │  memory check   │ ─────────────────────────────────────▶ DUPLICATE (suppressed)
 └────────┬────────┘
          ▼
 ┌─────────────────┐   p ≥ 0.60 → FORWARD  (+ type hint, probs, reason)
 │ 5. threshold    │   p ≤ 0.40 → REJECT
 │  gate           │   else     → UNCERTAIN (forwarded, low-priority flag)
 └─────────────────┘
```

Session state (all tiny, all in RAM): a 3-utterance context ring buffer
(reported downstream for explainability), the 12-slot duplicate memory of
forwarded vectors, and decision counters.

## 3. Why these components

- **Char n-grams** absorb ASR spelling drift and code-mixed tokens without a tokenizer file.
- **Logistic Regression over LinearSVC**: native `predict_proba` gives the calibrated scores the uncertainty band needs; measured slightly better on unseen templates anyway (0.722 vs 0.713 macro-F1).
- **Asymmetric thresholds (0.40 / 0.60)**: a false reject destroys a moment forever; a false forward wastes pennies of compute. The grey band converts would-be coin-flips into an explicit `UNCERTAIN` signal — the behaviour the brief asks for.
- **Duplicate memory on *forwarded* items only**: repeats of rejected chit-chat are harmless; repeats of forwarded moments spam downstream. Threshold 0.45 chosen from measurements: true restatements score ≥ 0.62 with this vectorizer, same-topic-but-different moments ≤ 0.18.
- **Classify the utterance alone**: concatenating prior context into the model input was tried and *removed* — the vectorizer was trained on single utterances, and context words from a previous meaningful line contaminated short ordinary lines (measured on the scenario suite). Context-aware features are future work, done properly (train-time change, not serve-time hack).

## 4. Repository layout

```
edge-gatekeeper/
├── main.py                 FastAPI app — wires everything together
├── requirements.txt
├── notebooks/
│   ├── 01_data_generation.ipynb            builds data/ (synthetic + scenarios)
│   └── 02_model_training_and_comparison.ipynb  trains, evaluates, joblib-dumps to models/
├── data/                   train.csv · test.csv · eval_scenarios.json
├── models/                 tfidf_vectorizer.joblib · gatekeeper_classifier.joblib
│                           model_meta.json · model_size_report.txt
├── backend/
│   ├── gatekeeper.py       GatekeeperEngine (the product)
│   ├── simulator.py        scenario replay + CLI demo
│   └── schemas.py          pydantic API models
├── frontend/               index.html · style.css · script.js (demo console)
└── docs/                   this documentation set
```