# Rating filter and developer overlap comparison

Implemented for the team's request to remove games rated below 3.5 and compare developer counts before/after across the existing chronological partition. No minimum rating-count filter is applied.

## View results

The audit is already generated. Double-click `artifacts/rating_developer_audit/index.html` for a standalone, searchable results page. It works without installing Python dependencies. CSV files alongside it open in Excel; they include every developer, not just the top studios.

To use the full existing dashboard on this computer, from this directory in PowerShell:

```powershell
.\.venv-local\Scripts\python.exe run_dashboard.py --no-debug
```

Open http://127.0.0.1:5000/data-audit or select **Rating & Developer Audit** in the sidebar. The other dashboard pages show existing model predictions, explanations, counterfactual suggestions, and historical benchmark tables. Those are not new scores for the filtered dataset.

## Main result: existing binary modeling cohort

| Measure | Before | After dropping known ratings below 3.5 |
| --- | ---: | ---: |
| Games | 12,565 | 11,790 |
| Developers | 7,071 | 6,731 |
| Releases through 2016 | 7,322 | 6,728 |
| Releases from 2017 | 5,243 | 5,062 |
| Developers in both periods | 490 | 466 |
| Later games from shared developers | 962 | 912 |
| Percentage of later games from shared developers | 18.35% | 18.02% |

775 games are removed (6.17% of the binary cohort). Low ratings account for 14.74% of games with a recorded rating in that cohort. The chat's approximate 18% should not be repeated as a verified percentage for this population.

There are 7,307 unrated games. The primary comparison retains them, because an absent rating does not mean a rating below 3.5. A sensitivity comparison also shows the stricter alternative: requiring a recorded rating ≥3.5 leaves 4,483 games, 171 shared developers, and 20.08% of later games from shared developers. Thus a simple `rating >= 3.5` implementation would also remove thousands of unrated games. This policy needs to be explicit in any final modeling change.

The all-store comparison is included separately: 17,007 → 15,944 games after removing 1,063 known low-rated games. Do not mix that denominator with the modeling cohort.

## Interpretation

Rating filtering barely changes the fraction of later games whose developer appears in training. It does not solve developer overlap. This audit measures overlap; it does not prove overlap increased prediction accuracy. A separate developer-disjoint evaluation would be required to measure generalization to unseen developers.

Time periods use **Original Release Date**, with training-era releases through 31 December 2016 and evaluation-era releases from 1 January 2017. These are audit group labels, not replacements for the saved random train/test parquet files. The source's later releases are from the historical dataset, not new 2026 data. Developer names are exact matches after trimming whitespace; aliases are not merged. The dataset field is Developer, not a separately verified publisher entity.

Filtering on player ratings narrows the study population and is not evidence of higher model accuracy or commercial success. Player ratings are post-launch observations, so they are a cohort-selection criterion, not an input Maya would know for a new game.

## Deliverables and reproduction

- `tools/rating_developer_audit.py`: standard-library-only reproducible analysis.
- `artifacts/rating_developer_audit/index.html`: searchable results and downloads.
- `summary.json`: cohort totals, explicit policies, source SHA-256, and every developer.
- `developer_comparison.csv`: before/after counts and period overlap for every binary-cohort developer.
- `cohort_comparison.csv`: overall comparison and unrated sensitivity analysis.
- `filtered_binary_games.csv`: filtered modeling cohort with original source columns.
- `filtered_all_games.csv`: corresponding filtered complete store dataset.

Regenerate and check:

```powershell
python tools/rating_developer_audit.py
python -m unittest discover -s tests -p test_rating_developer_audit.py -v
```

Raw CSVs, saved training partitions, models, and participant records are preserved. No model retraining or new accuracy claim is part of this task. All implementation changes are in PROJECT-V2/PROJECT; PROJECT is unchanged.
