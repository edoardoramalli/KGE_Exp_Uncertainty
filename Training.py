"""
Training.py
=================

Module for training knowledge graph embedding models on synthetic datasets
generated from experimental data scenarios using PyKEEN.

This script uses the PyKEEN `pipeline()` function to train models on the
binary dataset splits (training, validation, testing) created by
`generateDataSets.py` and stored under each scenario folder.

It supports configurable training parameters and optional offset tags
for experiment tracking.

Dependencies:
    - PyKEEN
    - OS

Example
-------
from Training import train
train('./Scenario/Case1', num=3, model='TransE', model_kwargs={'embedding_dim': 128})
"""

import os
from typing import Optional, Any
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


def train(
    path_scenario: str,
    num: int = 1,
    offset_x1: Optional[Any] = None,
    offset_x2: Optional[Any] = None,
    **kwargs
) -> None:
    """
    Train one or more knowledge graph embedding models on a scenario dataset.

    This function loads pre-generated dataset splits (training, validation, testing)
    and trains PyKEEN models using the `pipeline()` API. Each run can be customized
    via keyword arguments and will be saved under a scenario-specific directory.

    Parameters
    ----------
    path_scenario : str
        Path to the scenario folder containing the 'Dataset' directory.
    num : int, optional
        Number of model runs to perform. Default is 1.
    offset_x1 : Optional[Any], optional
        First offset tag for naming result directories. Useful for grid or parameter sweeps.
    offset_x2 : Optional[Any], optional
        Second offset tag for naming result directories.
    **kwargs
        Additional keyword arguments passed to `pykeen.pipeline.pipeline()`.
        These override the defaults.

    Default Configuration
    ---------------------
    - Model: RotatE
    - Embedding dimension: 64
    - Training loop: SLCWA
    - Epochs: 15,000
    - Batch size: 8192
    - Early stopping: enabled (frequency=500, patience=3, metric="hits@3")

    Output
    ------
    The trained models and logs are saved to directories such as:
    `path_scenario/results_<i>` or `path_scenario/results.<offsets>_<i>`
    """

    # --- Load Datasets -----------------------------------------------------
    training = TriplesFactory.from_path_binary(os.path.join(path_scenario, "Dataset", "training"))
    testing = TriplesFactory.from_path_binary(os.path.join(path_scenario, "Dataset", "testing"))
    validation = TriplesFactory.from_path_binary(os.path.join(path_scenario, "Dataset", "validation"))

    # --- Run Training Loops ------------------------------------------------
    for i in range(num):
        print(f"[INFO] Training run {i + 1}/{num}")

        # Default training configuration
        default_config = dict(
            training=training,
            testing=testing,
            validation=validation,
            model="RotatE",
            model_kwargs=dict(embedding_dim=64),
            training_loop="slcwa",
            training_kwargs=dict(num_epochs=15000, batch_size=2048 * 4),
            use_tqdm=True,
            stopper="early",
            stopper_kwargs=dict(
                frequency=500,
                patience=3,
                relative_delta=0.005,
                metric="hits@3",
            ),
        )

        # Merge with any user-provided overrides
        config = {**default_config, **kwargs}

        # --- Train model via PyKEEN pipeline ---
        result = pipeline(**config)

        # --- Construct output filename based on optional offsets ---
        if offset_x1 and offset_x2:
            file_name = f"results.{offset_x1}.{offset_x2}_{i}"
        elif offset_x1:
            file_name = f"results.{offset_x1}_{i}"
        else:
            file_name = f"results_{i}"

        output_dir = os.path.join(path_scenario, file_name)

        # --- Save results to directory ---
        result.save_to_directory(output_dir)
        print(f"[INFO] Training completed. Results saved at: {output_dir}")
