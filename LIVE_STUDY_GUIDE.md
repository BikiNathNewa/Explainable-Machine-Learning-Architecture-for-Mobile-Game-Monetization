# Live three-class participant study

Open http://127.0.0.1:5000/cards and choose Live Participant Experiment.
Practice first using Interactive Demo Mode if needed.

## Classes and scoring

- Class 0: no IAP, or maximum IAP below $4.99.
- Class 1: maximum IAP from $4.99 up to, but excluding, $19.99.
- Class 2: maximum IAP of $19.99 or more.

The live task predicts the catalog's recorded price class, not revenue or
advertising strategy. Treatment uses the separately trained, rating-filtered
chronological three-class XGBoost pipeline and actual per-class TreeSHAP
contributions. Model predictions may disagree with the catalog label.

## What a participant does

1. Enter an alias, role and experience; give consent.
2. Answer 12 shuffled profiles: six Control (mechanics only), six Treatment
   (mechanics plus model probabilities and explanations). Each condition has
   two actual examples of each class, with no repeated profile within a session.
3. Choose class 0, 1 or 2 and confidence from 1 to 5. Notes are optional.
4. Finish the session and download the answer CSV. Reloading the page offers
   Resume for the current session in that browser tab.

Practice uses three reserved real catalog profiles, one from each class. Votes
remain in the browser and are not saved. Practice profiles are excluded from
live assignment. Source game names and price labels are hidden during live
trials, and the current-card API does not return the actual class.

## Live study results

Pilot Study Results refreshes every five seconds, and immediately after a saved
answer in the same tab. The summary table, accuracy bars and class-level table
all use the same saved live trials. Partial sessions are included and labeled
in Participant Records. With no answers, the view shows zero trials and
“Not yet available,” with no fallback scores.

The server assigns the deck, reads actual labels, calculates correctness and
elapsed time, and records completion. Client-supplied scores, actual labels,
cohorts and timing are ignored. A duplicate submission of the same assigned
card does not add another trial. Elapsed time includes pauses between assignment
and submission and is not a precise measure of active attention.

## Storage and provenance

This study starts fresh in `artifacts/tables/live_study_v2.sqlite3`. SQLite
transactions and unique session/card keys keep live submissions consistent.
The CSV exports are generated from that database and include the study version,
model identifier and live source. Preserve the database when moving the project
if you want to keep participant results.

Historical binary CSV logs, fixed pilot scores, example participant quotes and
synthetic powered-trial artifacts are not imported or displayed on this page.
They remain in their original files for project history. They cannot be mapped
reliably into the new class numbering (high IAP is now class 2).

Session count is not a verified count of distinct people. Differences between
Control and Treatment are descriptive, not proof of a causal benefit. The
balanced card selection is a study design, not the store's natural class mix.

## Verification

Run `.venv-local/Scripts/python.exe -m pytest tests -q`. Participant tests use
temporary databases; never run simulated participants against the real store.
