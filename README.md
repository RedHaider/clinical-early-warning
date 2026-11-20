# Clinical Early Warning

An original synthetic temporal-risk benchmark exploring patient-state
prediction and therapy optimization.
It predicts deterioration from a short sequence of simulated vital signs and
compares a last-observation baseline with trend-aware features.

This project does not reproduce a professor's method and is not a clinical
decision-support product. It uses no patient records.

## Run

```powershell
python -m pip install -e .
python -m clinical_warning.experiment --output outputs/demo
pytest -q
```

## Evaluation

The experiment reports AUROC, average precision, Brier score, sensitivity, and
specificity. A subgroup report checks performance across two simulated age
groups to make model limitations visible.
