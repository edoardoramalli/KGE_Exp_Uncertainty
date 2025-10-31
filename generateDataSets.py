"""
generateDataSets.py

This script splits a single TSV knowledge graph dataset into
training, testing, and validation subsets using PyKEEN's TriplesFactory.

Expected directory structure:
    Scenario/<scenario_name>/Dataset/<dataset_name>.tsv

Outputs:
    Scenario/<scenario_name>/Dataset/training/
    Scenario/<scenario_name>/Dataset/testing/
    Scenario/<scenario_name>/Dataset/validation/

Each folder contains the respective binary PyKEEN dataset.
"""

import os
import glob
from pykeen.triples import TriplesFactory


def split_and_generate_dataset(path_scenario: str) -> None:
    """
    Split a TSV knowledge graph dataset into training, testing, and validation sets,
    and save each split in PyKEEN binary format.

    This function expects a folder structure like:
        path_scenario/
        └── Dataset/
            └── my_data.tsv

    The TSV file should contain triples in the format:
        head<TAB>relation<TAB>tail

    It performs the following steps:
        1. Finds the single TSV file under `<path_scenario>/Dataset/`.
        2. Loads it into a PyKEEN TriplesFactory.
        3. Randomly splits the dataset into 80% training, 10% testing, and 10% validation.
        4. Saves each split as a binary PyKEEN dataset for later use.

    Args:
        path_scenario (str): Path to the root scenario folder containing the `Dataset` subfolder.

    Raises:
        ValueError: If no TSV file or more than one TSV file is found in the Dataset folder.
    """
    # Locate the TSV dataset file in the given path
    tsv_files = glob.glob(os.path.join(path_scenario, 'Dataset', '*.tsv'))

    if len(tsv_files) == 0:
        raise ValueError(f"No TSV file found in '{os.path.join(path_scenario, 'Dataset')}'.")
    if len(tsv_files) > 1:
        raise ValueError(
            f"Multiple TSV files found in '{os.path.join(path_scenario, 'Dataset')}'; expected exactly one.")

    tsv_file = tsv_files[0]

    # Load triples into a TriplesFactory
    tf = TriplesFactory.from_path(tsv_file)

    # Split dataset into training (80%), testing (10%), and validation (10%)
    training, testing, validation = tf.split([0.8, 0.1, 0.1])

    # Define output paths for each split
    dataset_dir = os.path.join(path_scenario, 'Dataset')
    output_paths = {
        'training': os.path.join(dataset_dir, 'training'),
        'testing': os.path.join(dataset_dir, 'testing'),
        'validation': os.path.join(dataset_dir, 'validation'),
    }

    # Save the splits in binary PyKEEN format
    training.to_path_binary(path=output_paths['training'])
    testing.to_path_binary(path=output_paths['testing'])
    validation.to_path_binary(path=output_paths['validation'])

    print(f"Dataset successfully split and saved to:\n"
          f"  - {output_paths['training']}\n"
          f"  - {output_paths['testing']}\n"
          f"  - {output_paths['validation']}")
