#!/usr/bin/env bash
# End-to-end orchestration script executing all notebooks and verifying outputs
set -e

echo "Starting Explainable AI Game Monetization Pipeline Execution..."
echo "Timestamp: $(date)"

# Phase 1: Environment and Raw Data Verification
echo "[1/8] Running integration test suite on environment and schemas..."
python -m pytest tests/test_pipeline.py -v

# Phase 2: Notebook 01 - Ingestion and Labeling
echo "[2/8] Executing 01_data_ingestion_and_labeling.ipynb..."
python -m nbconvert --to notebook --execute --inplace notebooks/01_data_ingestion_and_labeling.ipynb

# Phase 3: Notebook 02 - Exploratory Data Analysis
echo "[3/8] Executing 02_exploratory_data_analysis.ipynb..."
python -m nbconvert --to notebook --execute --inplace notebooks/02_exploratory_data_analysis.ipynb

# Phase 4: Notebook 03 - Feature Engineering and Modeling
echo "[4/8] Executing 03_feature_engineering_and_modeling.ipynb..."
python -m nbconvert --to notebook --execute --inplace notebooks/03_feature_engineering_and_modeling.ipynb

# Phase 5: Notebook 04 - Faithfulness Evaluation
echo "[5/8] Executing 04_faithfulness_evaluation.ipynb..."
python -m nbconvert --to notebook --execute --inplace notebooks/04_faithfulness_evaluation.ipynb

# Phase 6: Notebook 05 - Stability Evaluation
echo "[6/8] Executing 05_stability_evaluation.ipynb..."
python -m nbconvert --to notebook --execute --inplace notebooks/05_stability_evaluation.ipynb

# Phase 7: Notebook 06 - Human Evidence Study
echo "[7/8] Executing 06_human_evidence_study.ipynb..."
python -m nbconvert --to notebook --execute --inplace notebooks/06_human_evidence_study.ipynb

# Phase 8: Notebook 07 - Actionable Counterfactuals
echo "[8/8] Executing 07_actionable_counterfactuals.ipynb..."
python -m nbconvert --to notebook --execute --inplace notebooks/07_actionable_counterfactuals.ipynb

# Final verification
echo "Running post-execution pipeline validation..."
python -m pytest tests/test_pipeline.py -v

echo "All 7 notebooks executed successfully with zero errors."
echo "All figures rendered to artifacts/figures/ and tables to artifacts/tables/."
