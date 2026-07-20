# Problem Understanding & Requirements

## 1. How I interpret the Gatekeeper problem

A wearable streams audio to a phone; speech-to-text turns it into transcript
fragments that arrive **incrementally, while people are still talking**. Almost
all of it is ordinary conversation. Somewhere inside it are rare moments —
questions, commitments, decisions, important updates, risks — that a more
capable (and more expensive) intelligence layer should look at.

The Gatekeeper is therefore a **router, not an assistant**. It answers exactly
one narrow question per fragment:

> *"Is this moment worth spending expensive intelligence on?"*

Three consequences follow from that framing:

1. **The gate decision is binary; the moment type is only a hint.**
   Downstream needs FORWARD vs REJECT to be right. Whether a forwarded moment
   was labelled `task` vs `decision` matters much less — the big model will
   re-read the text anyway. This shapes the evaluation: binary gate metrics
   are primary, fine-grained multiclass metrics are secondary.

2. **The two error types are asymmetric.**
   A falsely rejected moment is *gone forever* — the user never gets help.
   A falsely forwarded moment only wastes some downstream compute.
   The design must therefore bias toward recall of meaningful moments and use
   an explicit uncertainty band rather than force-classifying grey cases.

3. **"Cheap" is the whole point.**
   If the Gatekeeper itself were expensive, we would just send everything
   downstream. Every design choice must respect: ≤ 25 MB total model assets,
   CPU-only inference, millisecond-level per-fragment latency, no network.

## 2. Key technical challenges identified

| Challenge | Why it is hard | How this project addresses it |
|---|---|---|
| ASR-style input | No punctuation, lowercase, fillers, dysfluencies, code-mixed words ("arre", "yaar") | Training data is noised to match (see `docs/data_strategy.md`); char n-gram features tolerate spelling drift |
| Incremental operation | Can't wait for the full conversation | Stateless per-fragment classification + small rolling session state (context ring buffer, duplicate memory) |
| Repeats | People restate important things ("the stove is on" … "stove's still on, check it") | Cosine-similarity duplicate memory over recently forwarded moments, threshold picked from measured similarities |
| Uncertainty | Vague lines ("we should do something about it") shouldn't get a confident verdict | Calibrated probabilities + an explicit UNCERTAIN band (0.40–0.60) that forwards with a low-priority flag |
| No labelled data exists | Real ambient conversations are private | Template + slot synthetic generation, with a template-disjoint train/test split so evaluation measures generalisation, not memorisation |
| 25 MB budget | Rules out most transformer encoders once tokenizer + weights are counted | Sparse linear model: ~0.3 MB total, 1–3 ms/utterance on laptop CPU |

## 3. Functional requirements

- **FR1** Accept one transcript fragment at a time and return a decision before the next fragment arrives (≪ real-time speech rate).
- **FR2** Produce one of four decisions: `FORWARD`, `REJECT`, `UNCERTAIN`, `DUPLICATE`.
- **FR3** Attach a moment-type hint (`question | task | decision | info | risk | ordinary`) and a full probability distribution to every decision, so downstream can prioritise.
- **FR4** Never confidently classify inside the configured grey band; route those as `UNCERTAIN` instead.
- **FR5** Suppress near-repeats of moments already forwarded in the recent session window.
- **FR6** Run fully locally: no backend, cloud, remote LLM, or classification API at inference time.
- **FR7** Total model assets (weights + vectorizer vocabularies + metadata) ≤ 25 MB.
- **FR8** Expose the decision rationale (threshold comparison, similarity match) for debuggability.

## 4. Assumptions

- The speech-to-text layer segments audio into utterance-level fragments (roughly one speaker turn or sentence). Segmentation quality is out of scope.
- English-dominant conversation with light Indian-English code-mixing; other languages are future work.
- A "session" is one continuous conversation; duplicate memory and context reset between sessions.
- Downstream can tolerate some extra forwards (precision is negotiable) but lost moments are costly (recall is not).
- The laptop prototype's CPU latency is an upper-bound proxy for a modern phone's big cores; sparse linear inference ports to Android via ONNX or a ~200-line direct reimplementation (see `docs/architecture.md`).

## 5. Alternative approaches considered

| Approach | Size | Pros | Why not selected |
|---|---|---|---|
| Pure rules/keywords | ~0 MB | Transparent, instant | Macro-F1 0.44 on unseen phrasings (measured in notebook 02); brittle, no confidence |
| **TF-IDF + Logistic Regression (selected)** | **0.3 MB** | Native calibrated probabilities → clean uncertainty band; 1–3 ms; trivially portable | — |
| TF-IDF + LinearSVC | 0.3 MB | Strong sparse baseline | Slightly worse here (0.69 vs 0.73 macro-F1) and needs an extra calibration wrapper for probabilities |
| Small embedding model (MiniLM-class) + classifier head | 15–90 MB | Better semantics, paraphrase robustness | Eats most/all of the 25 MB budget once tokenizer + ONNX runtime are counted; 10–50× latency; overkill for short-utterance routing |
| Distilled tiny transformer trained from scratch | 5–20 MB | Could learn context | Needs far more (real) data than one week allows; synthetic data would just teach it the templates |
| Hybrid rules + ML (partially adopted) | 0.3 MB | Rules catch trivial cases for free | Adopted as a *pre-filter only*: a regex back-channel gate rejects pure fillers ("haha", "cool cool") without running ML |

The selected design is deliberately the smallest thing that meets every hard
constraint, with the uncertainty and duplicate machinery — the parts the
assignment emphasises — built around it. Upgrading the classifier later (e.g.
to a quantised MiniLM within the remaining 24.7 MB of budget) changes one
component without touching the gate logic.
