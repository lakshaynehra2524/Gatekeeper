# Edge Gatekeeper — Ambient Conversation Intelligence

AI/ML internship assignment: a lightweight, **fully on-device ML gate** that
watches incremental conversation transcripts and decides which moments deserve
deeper (expensive) intelligence — within a **25 MB model budget**.

**This solution uses 0.34 MB of that budget** and decides each fragment in
**1–3 ms on CPU**, with explicit uncertainty handling and duplicate
suppression.

```
transcript fragment ──▶ back-channel filter ──▶ TF-IDF + LogisticRegression
                        ──▶ duplicate memory ──▶ threshold gate
                        ──▶ FORWARD · REJECT · UNCERTAIN · DUPLICATE
```

## Quick start

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# open http://127.0.0.1:8000  → interactive demo console
```

Pre-trained models are committed in `models/`, so the server runs immediately.

**CLI simulator** (no browser needed):

```bash
python -m backend.simulator          # replay the evaluation conversations
python -m backend.simulator --live   # type your own utterances
```

**Retrain from scratch** (regenerates data + models):

```bash
cd notebooks
jupyter notebook   # run 01_data_generation.ipynb, then 02_model_training_and_comparison.ipynb
```

## Project structure

| path | purpose |
|---|---|
| `main.py` | FastAPI app connecting backend, models and frontend |
| `notebooks/01_data_generation.ipynb` | synthetic dataset + ASR noise + template-disjoint split → `data/` |
| `notebooks/02_model_training_and_comparison.ipynb` | compares rules / LogReg / LinearSVC, evaluates, `joblib`-dumps → `models/` |
| `data/` | `train.csv`, `test.csv`, `eval_scenarios.json` |
| `models/` | `tfidf_vectorizer.joblib`, `gatekeeper_classifier.joblib`, `model_meta.json`, `model_size_report.txt` |
| `backend/gatekeeper.py` | the Gatekeeper engine: thresholds, uncertainty band, duplicate memory, back-channel filter |
| `backend/simulator.py` | incremental conversation replay + scoring + CLI demo |
| `frontend/` | HTML/CSS/JS demo console (transcript stream, gate meter, probability inspector) |
| `docs/` | problem understanding · data strategy · architecture · evaluation report · limitations |

## API

| endpoint | description |
|---|---|
| `POST /api/evaluate` `{"text": "..."}` | gate one incremental fragment (stateful session) |
| `POST /api/reset` | new conversation session |
| `GET /api/model-info` | model size vs limit, thresholds, session stats |
| `GET /api/scenarios` | replay conversations |
| `GET /api/run-eval` | score all scenarios on a fresh engine |
| `GET /docs` | interactive OpenAPI docs |

## Headline results

| metric | value |
|---|---|
| combined model assets | **0.340 MB** (limit: 25 MB) |
| latency per fragment (laptop CPU) | **~1.4 ms** |
| true-drop rate at deployed thresholds, template-unseen test set | **0.00%** |
| false-reject rate, naive 0.5 cut | 0.21% |
| forward precision @ recall 0.969 | 0.986 (threshold 0.71) |
| macro-F1, 6-class, template-unseen | 0.722 (rule baseline: 0.444) |
| scenario suite (whole engine, hand-written) | 20/22 turns |

Full analysis including remaining scenario failures, the hedge-aware training
augmentation, and both error-cost directions: `docs/evaluation_report.md`.

## Submission statement (per assignment §10)

- **Implemented:** synthetic data pipeline (incl. a targeted hedge/vague-
  referent training augmentation, `notebooks/01_data_generation.ipynb` §4b),
  model comparison + training, gate engine (thresholds, uncertainty band,
  duplicate suppression, back-channel filter, session state), incremental
  simulator, FastAPI service, browser demo console, full evaluation with
  failure analysis.
- **Conceptual only:** Android/Kotlin port (path detailed in
  `docs/architecture.md` §5), real-data annotation loop
  (`docs/data_strategy.md` §5), second-stage semantic re-scorer
  (`docs/limitations_future_work.md`).
- **External libraries/models:** scikit-learn, joblib, numpy, pandas, FastAPI,
  uvicorn (all permissive licenses: BSD-3/MIT/Apache-2.0). **No pre-trained
  models or external datasets were used** — the classifier is trained from
  scratch on generated data; all data/label design is original work.
- **Total combined model size:** 0.340 MB (see `models/model_size_report.txt`).
- **Edge deployment:** pure-CPU sparse linear inference; port via direct
  Kotlin reimplementation (<1 MB assets) or ONNX Runtime Mobile —
  `docs/architecture.md` §5.
