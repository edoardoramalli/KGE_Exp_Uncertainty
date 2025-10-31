# Knowledge graph embedding for experimental uncertainty estimation

This repository provides a skeleton of the code and data used for the paper [Knowledge graph embedding for experimental uncertainty estimation](https://doi.org/10.1108/IDD-06-2022-0060).

---

## Citation
Ramalli E, Pernici B (2023), "Knowledge graph embedding for experimental uncertainty estimation". Information Discovery and Delivery, Vol. 51 No. 4 pp. 371–383, doi: https://doi.org/10.1108/IDD-06-2022-0060

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
  - [Dataset Generation](#dataset-generation)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
  - [Analysis](#analysis)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project enables you to:

1.  **Generate synthetic datasets** that simulate experiments with attributes such as type, reactor, author, year, target, similarity, and uncertainty.
2.  Convert these datasets into a **knowledge graph** format.
3.  Train **knowledge graph embedding models** (e.g., `RotatE`) on the generated datasets.
4.  **Evaluate model predictions** using rank-based metrics and calculate differences for uncertainty predictions.
5.  **Analyze correlations** and other statistics within the datasets.

---

## Repository Structure

```
.
├── data/
├── BaseClasses.py              # Core classes: Triple, Experiment
├── generateDataSets.py         # Dataset splitting and binary generation
├── AnalyzeDataset.py           # Dataset analysis and correlation heatmaps
├── Training.py                 # Model training logic
├── evaluate.py                 # Model evaluation logic
├── read_json_evaluation_result.py # Aggregate evaluation results
├── pipeline.py                     # Full workflow orchestration
├── Scenario/                         # Folder for scenario datasets
└── README.md                       # This file
```

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repo_url>
    cd <repo_name>
    ```

2.  **Create and activate a Python environment** (Conda is recommended):
    ```bash
    conda create -n kg_pipeline python=3.10
    conda activate kg_pipeline
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include: `pykeen`, `torch`, `pandas`, `numpy`, `scipy`, `seaborn`, `matplotlib`, and `tqdm`.

4.  **Set the environment variable** for the computation device (e.g., `cpu` or `cuda`):
    ```bash
    export PYTHON_DEVICE=cpu
    ```

---

## Usage

The main workflow is orchestrated by `pipeline.py`. To run the pipeline for all scenarios matching the default pattern, execute:

```bash
python pipeline.py
```

You can also customize the scenario pattern and the number of training runs directly in your Python code:

```python
from pipeline import run_pipeline

run_pipeline(scenario_pattern='./Scenario/CaseI_21_*', train_runs=3)
```

---

## Workflow

### Dataset Generation

-   Converts experiments into triples in the format (`head`, `relation`, `tail`).
-   Splits the data into training, validation, and testing sets.

Each experiment includes the following attributes:
-   Experiment Type
-   Reactor
-   Author
-   Year
-   Target
-   Similarity
-   Uncertainty

The system can also track "close" experiments with similar attributes.

**Example:**

```python
from src.BaseClasses import Experiment

# Create a new experiment with ID 1
exp = Experiment(1)

# Print the experiment record and its knowledge graph representation
print(exp.to_record())
print(exp.to_kg())
```

### Model Training

-   Loads the training triples and trains a knowledge graph embedding model.
-   The default configuration uses the **RotatE** model with an embedding dimension of `64`, the `slcwa` training loop, and `15000` epochs.
-   Trained models are saved in directories specific to each scenario.

**Example:**

```python
from src.Training import train

# Train a model with a different embedding dimension
train(
    path_scenario='./Scenario/CaseI_21_0',
    num=3,
    model_kwargs={'embedding_dim': 128}
)
```

### Evaluation

-   Evaluates predictions for the `has_uncertainty` relation.
-   Computes the ranks of the true values and the difference between predicted and true uncertainty.
-   Stores the results in a JSON file.

**Example JSON Output:**

```json
{
  "U": ["U_0", "U_0.1", ...],
  "R": [1, 3, 5, ...],
  "D": [0.0, 0.1, -0.2, ...]
}
```

### Analysis

-   `AnalyzeDataset.py` can be used to generate correlation heatmaps and compute dataset statistics.
-   `read_json_evaluation_result.py` aggregates evaluation metrics from the JSON output files.

