# Data & Annotation Strategy

## 1. Why synthetic data

Real ambient-conversation transcripts are (a) private by nature, (b) not
publicly available in labelled form for this exact task, and (c) impossible to
collect + annotate responsibly within one week. The assignment explicitly
permits **provided, collected, or simulated data**, so this project simulates.

The risk with synthetic data is self-deception: a model can look perfect
because it memorised the generator, not the task. Two design decisions guard
against that (section 4).

## 2. Label taxonomy

Six classes, mapped 1:1 to the "meaningful moments" listed in the assignment
brief plus one negative class:

| label | assignment moment type | gate policy |
|---|---|---|
| `question` | questions that may need assistance | forward |
| `task` | tasks, commitments, follow-ups | forward |
| `decision` | decisions or changing plans | forward |
| `info` | important information or updates | forward |
| `risk` | warnings, risks, unusual situations | forward |
| `ordinary` | (everything else: chit-chat, fillers) | reject |

The taxonomy is deliberately coarse. The Gatekeeper's job is routing; finer
intent taxonomies (30+ intents) belong downstream where a capable model reads
the forwarded text anyway.

## 3. Generation method (notebook `01_data_generation.ipynb`)

**Template + slot filling.** Each class has 30–40 hand-written utterance
templates containing slots (`{person}`, `{place}`, `{thing}`, `{day}`,
`{hazard}`, …) filled from vocabulary pools, giving thousands of distinct
surface forms per class (1,400 sampled per class → 8,400 total).

**ASR-style noising.** Wearable speech-to-text output does not look like
written text. Every generated utterance is passed through a noise layer:

- lowercased, punctuation stripped (ASR rarely emits it),
- spoken fillers prepended/appended with probability ("um", "arre", "you know", "yaar"),
- occasional word repetition to simulate dysfluency ("we we should go").

This keeps the training distribution close to what the deployed model sees,
and it is why the runtime normalises input the same way before inference.

**Domain flavour.** Vocabulary is everyday Indian-household/office English
(bills, metro, biryani, monsoon, "acha okay") — matching a realistic first
deployment population and providing natural hard cases (code-mixed fillers).

## 4. Split design: the part that matters most

A naive random split leaks templates across train/test; every model then
scores ~100% and the evaluation is meaningless (this was observed directly
during development). Instead:

**Template-disjoint split** — ~20% of the *templates* in each class are held
out entirely; every test utterance comes from a phrasing the model has never
seen in any variation. This measures generalisation to new wordings, which is
the actual deployment condition.

**Hand-written scenario suite** (`data/eval_scenarios.json`) — 4 conversations
/ 22 turns written manually, *not* generated from templates, covering: clear
forwards, clear rejects, deliberate near-duplicate restatements, and
deliberately ambiguous lines where the correct behaviour is `UNCERTAIN`. This
suite evaluates the whole engine (thresholds, duplicate memory, back-channel
filter), not just the classifier.

## 5. Annotation strategy for real data (production path)

When real data becomes available, the plan is:

1. **Bootstrap with the synthetic model** — run it over consented pilot transcripts; its FORWARD/UNCERTAIN stream is the candidate pool.
2. **Annotate decisions, not transcripts.** Raters see one fragment + 3 previous lines of context and answer two questions: "should this reach the assistant?" (binary — the metric that matters) and "which moment type?" (multi-choice). Fragment-level binary labelling is fast (~4–6 s/item) and matches the model's task exactly.
3. **Prioritise disagreement.** Active learning: label items the current model puts in the uncertainty band first — cheapest accuracy gains per label.
4. **Measure inter-annotator agreement** on the binary question; items with low agreement are legitimately ambiguous and become UNCERTAIN training targets rather than being forced into a class.
5. **Privacy:** all annotation on opt-in pilot data, fragments only (no full-conversation export), PII scrubbing before storage; on-device personalisation (threshold tuning per user) never exports raw text.

## 6. Files produced

| file | contents |
|---|---|
| `data/train.csv` | 6,901 utterances: template-seen portion (6,702) + a train-only hedge/vague-referent augmentation (§4b of `notebooks/01_data_generation.ipynb`, added after the split below, see `docs/evaluation_report.md` §1/§3) |
| `data/test.csv` | 1,698 noised utterances, template-UNSEEN portion (untouched by the augmentation above) |
| `data/eval_scenarios.json` | 4 hand-written conversations, 22 turns with expected gate decisions |
