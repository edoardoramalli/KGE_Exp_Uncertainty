"""
evaluate.py
=================

Module for evaluating trained knowledge graph embedding models
on synthetic experimental datasets using PyKEEN.

This script loads trained models from disk, performs tail entity prediction
for `has_uncertainty` relations, and computes ranking-based metrics to
assess model performance.

The evaluation results (ranks and numerical differences between predicted
and true uncertainties) are stored as JSON files for each trained model.

Dependencies:
    - PyKEEN
    - PyTorch
    - NumPy
    - JSON
    - Glob
    - OS

"""

import os
import glob
import json
import torch
from typing import List, Dict
from pykeen.triples import TriplesFactory
from pykeen.models import predict

# Select computation device from environment variable
PYTHON_DEVICE = os.environ.get("PYTHON_DEVICE", "cpu")


def evaluate_result(current_path: str, training_object: TriplesFactory, testing_object: TriplesFactory) -> None:
    """
    Evaluate a trained model on the test dataset.

    This function loads a trained PyKEEN model from the given path,
    performs tail entity prediction for all triples with the relation
    'has_uncertainty' in the testing set, and computes:

        - The rank of the true uncertainty value in the predicted list.
        - The numerical difference between the top-1 predicted uncertainty
          and the true one.

    Results are saved as a JSON file in the same directory.

    Parameters
    ----------
    current_path : str
        Path to the folder containing the trained model (`trained_model.pkl`).
    training_object : TriplesFactory
        PyKEEN TriplesFactory for the training dataset.
    testing_object : TriplesFactory
        PyKEEN TriplesFactory for the testing dataset.
    """

    # Extract all uncertainty labels from the training set
    uncertainty_labels = [
        e for e in training_object.entity_labeling.label_to_id if e.startswith("U")
    ]

    # Load trained model
    model_path = os.path.join(current_path, "trained_model.pkl")
    model = torch.load(model_path, map_location=torch.device(PYTHON_DEVICE))

    results = []
    diffs = []

    # Evaluate only triples with the 'has_uncertainty' relation
    for triple in testing_object.triples:
        if triple[1] != "has_uncertainty":
            continue

        # Predict possible tail entities for (head, relation)
        predicted_tails_df = predict.get_tail_prediction_df(
            model,
            head_label=triple[0],
            relation_label="has_uncertainty",
            triples_factory=training_object,
        )

        # Filter predictions to only include uncertainty labels
        predicted_tails_df = predicted_tails_df[
            predicted_tails_df["tail_label"].isin(uncertainty_labels)
        ].reset_index(drop=True)

        # Rank of the correct uncertainty
        rank = int(predicted_tails_df[predicted_tails_df["tail_label"] == triple[2]].index.values[0])

        # Compute numeric difference between top-1 prediction and true uncertainty
        top_pred = float(predicted_tails_df.iloc[0]["tail_label"][2:])
        true_pred = float(predicted_tails_df.iloc[rank]["tail_label"][2:])
        diff = top_pred - true_pred

        diffs.append(diff)
        results.append(rank + 1)  # ranks start from 1

    # Save results to JSON
    evaluation_dict: Dict[str, List] = {
        "U": uncertainty_labels,
        "R": results,
        "D": diffs,
    }

    output_path = os.path.join(current_path, "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_dict, f, indent=4)

    print(f"[INFO] Evaluation completed and saved at: {output_path}")


def start_evaluate(path_scenario: str) -> None:
    """
    Run evaluation for all trained models in a scenario.

    Loads the training, testing, and validation datasets from the given scenario path,
    finds all trained model files (`.pkl`), and evaluates each using `evaluate_result()`.

    Parameters
    ----------
    path_scenario : str
        Base path of the scenario containing the dataset and trained models.

    Example
    -------
    start_evaluate('./Scenario/Case1')
    """

    # Load datasets
    training = TriplesFactory.from_path_binary(os.path.join(path_scenario, "Dataset", "training"))
    testing = TriplesFactory.from_path_binary(os.path.join(path_scenario, "Dataset", "testing"))
    validation = TriplesFactory.from_path_binary(os.path.join(path_scenario, "Dataset", "validation"))

    # Find all model checkpoints
    model_paths = glob.glob(os.path.join(path_scenario, "./**/*.pkl"), recursive=True)

    if not model_paths:
        print(f"[WARNING] No .pkl models found in {path_scenario}")
        return

    # Evaluate each model found
    for index, model_path in enumerate(model_paths, start=1):
        print(f"[INFO] Starting evaluation {index}/{len(model_paths)} for model: {model_path}")
        model_dir = os.path.dirname(model_path)
        evaluate_result(current_path=model_dir, training_object=training, testing_object=testing)
