# Limitations & Future Work

## Known limitations (where this approach fails)

**1. Lexical, not semantic.** TF-IDF sees words, not meaning. Paraphrases with
unusual vocabulary can slip below the forward threshold, and meaningful-
*sounding* vague lines ("we should do something about it") get overconfident
forwards. The scenario suite deliberately exposes this (see
`docs/evaluation_report.md`, failure modes 1–2).

**2. Trained on synthetic data.** The template-disjoint split proves
generalisation to unseen *phrasings*, but not to unseen *domains* or genuinely
spontaneous speech. Real-world accuracy will be lower until the model is
fine-tuned on annotated pilot data (plan in `docs/data_strategy.md` §5).

**3. Single-utterance decisions.** The engine tracks context but does not feed
it to the classifier (a serve-time context hack was tried and measurably hurt —
documented in `docs/architecture.md` §3). Moments that only become meaningful
across turns ("okay" — agreeing to a plan proposed two turns ago) are missed.

**4. Duplicate memory is session-bounded and lexical.** It remembers the last
12 forwarded moments by vector similarity; a repeat 30 minutes later, or a
semantic repeat with disjoint vocabulary ("the hob is on" vs "the stove is
on"), gets forwarded again. Also, "already-handled" suppression (downstream
resolved the moment, stop forwarding follow-ups) requires a feedback channel
that does not exist yet.

**5. English + light code-mixing only.** Full Hindi/Hinglish conversations are
out of scope for this week; the char n-gram design was chosen partly because it
extends naturally, but data is needed.

**6. No speaker awareness.** TV audio, other people's conversations, and the
user's own speech are treated identically. Diarisation is explicitly out of
scope in the brief, but a production gate would want at least an
"is-user-involved" bit.

**7. Thresholds are global.** 0.40/0.60 were chosen from the precision-recall
sweep on synthetic data; different users have different tolerance for
over-forwarding. No per-user adaptation exists yet.

## Future work (ordered by value/effort)

1. **Annotated pilot data + active learning loop** — label the model's own
   UNCERTAIN band first; retrain. Biggest accuracy win, no architecture change.
2. **Context-aware features done properly** — train the classifier on
   (previous-turn, current-turn) pairs so context helps instead of
   contaminating; targets failure modes 1 and 3.
3. **Downstream feedback channel** — let the intelligence layer mark moments as
   handled; extend duplicate suppression to "already-handled" suppression.
4. **Kotlin port + on-device benchmark** — ~200-line reimplementation
   (architecture doc §5), measure latency/battery on a mid-range phone.
5. **Semantic upgrade within budget** — a quantised MiniLM-class encoder
   (~15–20 MB int8) as a *second-stage* re-scorer for the uncertainty band
   only: the cheap linear model still handles the confident 90%+, the encoder
   only runs on grey-band items. Stays under 25 MB, fixes paraphrase blindness
   where it matters most.
6. **Per-user threshold adaptation** — adjust the forward threshold from
   implicit feedback (dismissed vs engaged downstream results).
7. **Hinglish expansion** — extend templates + vocabulary, verify char n-grams
   carry over, then collect real code-mixed pilot data.
