#!/usr/bin/env python3
"""Print the pre-2020 XGBoost model-selection table on its own.

run_project.py already runs this step and saves results/model_selection.csv;
this script is for looking at the selection without the rest of the pipeline.

Purged schemes (P1, P2) leave a gap of at least the 365-day label window
between the last fit date and the first validation date, so no inspection
labels both a fit row and a validation row. Only they enter the criterion.
The unpurged schemes (U1, U2) are shown for reference: consecutive quarterly
rows share the same next inspection, so in those schemes up to ~40% of
validation labels were already seen in fitting.
"""
from food_inspections import config as C
from food_inspections.data import clean, load_raw
from food_inspections.features import load_panel
from food_inspections.models import design_matrix, select_xgboost


def main():
    panel = load_panel(clean(load_raw()))
    train = panel[panel["period"] == "train"]
    selected, table = select_xgboost(train, design_matrix(panel)[panel["period"] == "train"])
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(C.RESULTS_DIR / "model_selection.csv", index_label="config")
    print(table.round(3).to_string())
    print(f"\nSELECTED: {selected}")


if __name__ == "__main__":
    main()
