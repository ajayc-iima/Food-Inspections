#!/usr/bin/env python3
"""Which restaurants should be inspected first?

Runs the whole project on the City of Chicago Food Inspections data:
download (or reuse the cached snapshot) -> clean -> point-in-time panel ->
pre-2020 model selection -> baseline / logistic / XGBoost -> Precision@K on
2022-2025 -> results/ and charts/.

    python run_project.py                  # uses data/ caches
    python run_project.py --rebuild-panel  # rebuild the panel from the raw snapshot
    python run_project.py --refresh        # re-download the live dataset first
"""
import argparse

from food_inspections.pipeline import run


def main():
    ap = argparse.ArgumentParser(description="Chicago food-inspection prioritisation")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download the dataset (also rebuilds the panel)")
    ap.add_argument("--rebuild-panel", action="store_true",
                    help="rebuild data/panel_examples.csv.gz from the raw snapshot")
    args = ap.parse_args()
    run(refresh=args.refresh, rebuild_panel=args.rebuild_panel)


if __name__ == "__main__":
    main()
