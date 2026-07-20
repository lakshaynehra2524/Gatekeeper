# Experiment & Evaluation Report

All numbers below are produced by the committed code: dataset metrics come from
`notebooks/02_model_training_and_comparison.ipynb` (executed, outputs saved) and the
scenario results from `backend/simulator.py` (`python -m backend.simulator`).

## 1. Classifier evaluation (template-UNSEEN test set)

Test set: 1698 utterances whose templates were fully held out of training —
the model has never seen these phrasings in any slot-filled variation.

| metric | value |
|---|---|
| macro-F1, 6-class (selected LogReg) | **0.733** |
| macro-F1, rule baseline | 0.444 |
| false-reject rate (useful moment confidently lost) | **0.21%** |
| false-forward rate at naive 0.5 cut | 16.73% |
| per-utterance latency (vectorize + predict, laptop CPU) | 1.94 ms |

Reading these together: the **binary gate** — the decision that matters — is strong
(0.2% of meaningful moments confidently lost; forward precision 0.994 at recall 0.976
at threshold 0.75 per the sweep in notebook 02). The lower multiclass score reflects
confusion **between meaningful classes** (decision ↔ info ↔ task) on never-seen phrasings;
since the type label is only a downstream hint, this costs little. The false-forward rate
at a naive 0.5 cut looks high, but most of those items fall inside the runtime uncertainty
band (0.40–0.60) and surface as `UNCERTAIN`, not as confident forwards.

## 2. Whole-engine evaluation (hand-written scenario suite)

22 turns across 4 conversations written independently of the training templates,
with an expected gate decision per turn. Replay: `python -m backend.simulator`.

**Overall: 18/22 turns matched the expected decision.**

### Morning at home — 6/7
*Household chatter with a safety risk, a task and a repeated line.*

| utterance | expected | got | p(meaningful) | ok |
|---|---|---|---|---|
| good morning | REJECT | REJECT | 0.03 | ✅ |
| slept okay i guess | REJECT | UNCERTAIN | 0.41 | ❌ |
| hey i think the stove is still on | FORWARD | FORWARD | 1.00 | ✅ |
| remind me to pay the electricity bill today | FORWARD | FORWARD | 1.00 | ✅ |
| the stove is still on someone check | DUPLICATE | DUPLICATE | 1.00 | ✅ |
| it's so hot these days | REJECT | REJECT | 0.04 | ✅ |
| mom said the school is closed tomorrow | FORWARD | FORWARD | 1.00 | ✅ |

### Office corridor — 6/6
*Work small talk that turns into a decision and a commitment.*

| utterance | expected | got | p(meaningful) | ok |
|---|---|---|---|---|
| did you watch the match yesterday | REJECT | REJECT | 0.01 | ✅ |
| haha yeah what a game | REJECT | REJECT | 0.01 | ✅ |
| okay so let's move the client meeting to thursday | FORWARD | FORWARD | 1.00 | ✅ |
| i'll send the revised deck by tonight | FORWARD | FORWARD | 1.00 | ✅ |
| cool cool | REJECT | REJECT | 0.00 | ✅ |
| also fyi the office is shifting to the new building next month | FORWARD | FORWARD | 1.00 | ✅ |

### Ambiguous moments — 1/4
*Lines where a confident decision is NOT desirable — the Gatekeeper should be uncertain.*

| utterance | expected | got | p(meaningful) | ok |
|---|---|---|---|---|
| we should do something about it | UNCERTAIN | FORWARD | 0.96 | ❌ |
| that might be a problem later | UNCERTAIN | FORWARD | 0.68 | ❌ |
| hmm i'm not sure that works | UNCERTAIN | REJECT | 0.16 | ❌ |
| yeah maybe | REJECT | REJECT | 0.01 | ✅ |

### Evening plans — 5/5
*Plan changes, a question, and repeated confirmations.*

| utterance | expected | got | p(meaningful) | ok |
|---|---|---|---|---|
| so tired today man | REJECT | REJECT | 0.01 | ✅ |
| do you know what time the pharmacy closes | FORWARD | FORWARD | 1.00 | ✅ |
| change of plan we're meeting at the cafe now | FORWARD | FORWARD | 1.00 | ✅ |
| we're meeting at the cafe now okay | DUPLICATE | DUPLICATE | 1.00 | ✅ |
| okay bye see you | REJECT | REJECT | 0.01 | ✅ |

## 3. Discussion of failures (required by the brief)

**Successes worth noting.** Duplicate suppression catches restatements that share almost
no exact wording ("hey i think the stove is still on" → "the stove is still on someone
check"); the back-channel filter rejects pure fillers in ~0 ms without running the model;
all clear forwards/rejects are decided correctly with high margins.

**Failure mode 1 — overconfident on vague meaningful-looking lines.** "we should do
something about it" and "that might be a problem later" get FORWARD (p≈0.96 / 0.68)
where UNCERTAIN was intended. Lexical features see strong cue words ("we should",
"problem") but cannot see that the *referent is unresolved*. Note the error is in the
safe direction — the moments are still forwarded, just without the low-priority flag.

**Failure mode 2 — hedged negation.** "hmm i'm not sure that works" is rejected
(p=0.16): hedging tokens dominate and the model has no representation of disagreement-
about-a-plan as meaningful. A genuine miss; the mitigation path (context features,
hedge-aware training data) is in `docs/limitations_future_work.md`.

**Failure mode 3 — borderline chit-chat lands in the band.** "slept okay i guess"
gets UNCERTAIN (p=0.41), i.e. forwarded low-priority instead of rejected. Cost: a few
wasted downstream tokens. This is the asymmetry working as designed — the band exists
to absorb exactly these, and the priced-in cost is stated below.

## 4. Error-cost analysis

**Too much forwarded (false forward / uncertain spam):** wastes downstream compute and
battery, and if the assistant surfaces trivia, erodes user trust. Bounded by: the
back-channel filter, duplicate suppression, and the 0.60 confident-forward threshold —
and UNCERTAIN items are flagged so downstream can process them at lower priority or in
batches.

**Useful moment rejected (false reject):** the moment is unrecoverable — an assistant
that misses "the stove is on" once loses more trust than one that occasionally over-
forwards. This is why thresholds are recall-biased and why the confident false-reject
rate (0.2% on unseen phrasings) is the primary safety metric.

## 5. Constraint compliance

```
MODEL SIZE REPORT — Edge Gatekeeper
=============================================
gatekeeper_classifier.joblib           0.230 MB
model_meta.json                        0.001 MB
tfidf_vectorizer.joblib                0.074 MB
---------------------------------------------
TOTAL (all model assets)               0.304 MB
Assignment limit                      25.000 MB
Headroom                              24.696 MB

Notes:
- No separate tokenizer/vocabulary files: the TF-IDF vocabularies
  are embedded inside tfidf_vectorizer.joblib and counted above.
- No quantisation or pruning was needed to meet the limit.
```
Incremental operation: per-fragment decision in ~1.94 ms
on a laptop CPU (measured in notebook 02; scenario replay shows 1–4 ms/turn including the
duplicate check) — orders of magnitude faster than speech arrives. No network calls exist
anywhere in the inference path.