"""
pipeline.py
===========

Orchestrates the full knowledge graph embedding workflow for multiple scenarios:
1. Splitting and generating datasets from TSV files.
2. Training embedding models using PyKEEN.
3. Evaluating trained models and storing results.

Dependencies:
    - Training.py
    - Evaluate.py
    - generateDataSets.py
    - glob
    - tqdm
"""

import glob
from tqdm import tqdm

from Training import train
from Evaluate import start_evaluate
from generateDataSets import split_and_generate_dataset


def run_pipeline(scenario_pattern: str = './Scenario/CaseI_21_*', train_runs: int = 3) -> None:
    """
    Execute the full pipeline on multiple scenarios matching a glob pattern.

    Parameters
    ----------
    scenario_pattern : str, optional
        Glob pattern to locate scenario directories. Default is './Scenario/CaseI_21_*'.
    train_runs : int, optional
        Number of training runs per scenario. Default is 3.
    """
    # Find all matching scenario directories
    paths = glob.glob(scenario_pattern)

    for scenario_path in tqdm(paths, desc="Running pipeline"):
        print('Processing scenario:', scenario_path)

        # Step 1: Generate and split dataset
        split_and_generate_dataset(path_scenario=scenario_path)

        # Step 2: Train models
        train(path_scenario=scenario_path, num=train_runs)

        # Step 3: Evaluate trained models
        start_evaluate(path_scenario=scenario_path)


if __name__ == "__main__":
    # Default execution
    run_pipeline()
