@echo off
REM End-to-end execution script for Windows environments
echo Starting Explainable AI Game Monetization Pipeline Execution...

set PYTHON_CMD=python
py -3.11 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py -3.11
)

echo [1/8] Running integration tests...
%PYTHON_CMD% -m pytest tests\test_pipeline.py -v
if %ERRORLEVEL% NEQ 0 (
    echo Integration tests failed! Exiting...
    exit /b %ERRORLEVEL%
)

echo [2/8] Executing 01_data_ingestion_and_labeling.ipynb...
%PYTHON_CMD% -m nbconvert --to notebook --execute --inplace notebooks\01_data_ingestion_and_labeling.ipynb
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [3/8] Executing 02_exploratory_data_analysis.ipynb...
%PYTHON_CMD% -m nbconvert --to notebook --execute --inplace notebooks\02_exploratory_data_analysis.ipynb
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [4/8] Executing 03_feature_engineering_and_modeling.ipynb...
%PYTHON_CMD% -m nbconvert --to notebook --execute --inplace notebooks\03_feature_engineering_and_modeling.ipynb
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [5/8] Executing 04_faithfulness_evaluation.ipynb...
%PYTHON_CMD% -m nbconvert --to notebook --execute --inplace notebooks\04_faithfulness_evaluation.ipynb
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [6/8] Executing 05_stability_evaluation.ipynb...
%PYTHON_CMD% -m nbconvert --to notebook --execute --inplace notebooks\05_stability_evaluation.ipynb
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [7/8] Executing 06_human_evidence_study.ipynb...
%PYTHON_CMD% -m nbconvert --to notebook --execute --inplace notebooks\06_human_evidence_study.ipynb
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [8/8] Executing 07_actionable_counterfactuals.ipynb...
%PYTHON_CMD% -m nbconvert --to notebook --execute --inplace notebooks\07_actionable_counterfactuals.ipynb
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo Running final validation...
%PYTHON_CMD% -m pytest tests\test_pipeline.py -v
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo All 7 notebooks executed successfully with zero errors.
echo All artifacts generated into artifacts\figures\ and artifacts\tables\.
