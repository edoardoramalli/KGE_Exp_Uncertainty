"""
This script analyzes a knowledge graph and its associated tabular dataset
to provide insights into entity/relation distribution and feature correlations.

It performs:
    1. Entity and relation counting from the TSV knowledge graph file.
    2. Categorical and numerical correlation analysis from a CSV file.
    3. Visualization of the resulting correlation matrix using seaborn.

Expected directory structure:
    Scenario/<scenario_name>/Dataset/<scenario_name>.tsv
    Scenario/<scenario_name>/Dataset/<scenario_name>_df.csv
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pykeen.triples import TriplesFactory
from scipy.stats import pearsonr, pointbiserialr, chi2_contingency


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

scenario = 'CaseG_21_2'
general_path = f'./Scenario/{scenario}/Dataset/'

# Input paths
triples_path = os.path.join(general_path, f'{scenario}.tsv')
df_path = os.path.join(general_path, f'{scenario}_df.csv')

# Rules for identifying entity types based on string prefixes
rules = [
    {'entity': 'experiment', 'rule': 'ID:'},
    {'entity': 'type', 'rule': 'TY_'},
    {'entity': 'reactor', 'rule': 'R_'},
    {'entity': 'author', 'rule': 'Author'},
    {'entity': 'year', 'rule': 'Y_'},
    {'entity': 'target', 'rule': 'T_'},
    {'entity': 'similarity', 'rule': 'S_'},
    {'entity': 'uncertainty', 'rule': 'U_'},
]


# -----------------------------------------------------------------------------
# KNOWLEDGE GRAPH PARSING
# -----------------------------------------------------------------------------

def check_entity(entity: str, list_rules: list[dict]) -> str:
    """
    Determine the entity type based on defined prefix rules.

    Args:
        entity (str): Entity label (e.g., 'R_1', 'Author_JaneDoe').
        list_rules (list[dict]): Mapping of prefix rules to entity types.

    Returns:
        str: The entity type (e.g., 'reactor', 'author').

    Raises:
        ValueError: If no matching rule is found.
    """
    for rule in list_rules:
        if entity.startswith(rule['rule']):
            return rule['entity']
    raise ValueError(f"No entity type found for '{entity}'.")


# Parse the triples file
entities = {}
relations = {}
triples_count = 0

with open(triples_path, 'r') as f:
    for line in f:
        triples_count += 1
        h, r, t = [x.strip() for x in line.split('\t')]

        type_h = check_entity(h, rules)
        type_t = check_entity(t, rules)

        # Accumulate entities per type
        entities[type_h] = entities.get(type_h, []) + [h]
        entities[type_t] = entities.get(type_t, []) + [t]

        # Count relation occurrences
        relations[r] = relations.get(r, 0) + 1

# -----------------------------------------------------------------------------
# PRINT BASIC STATISTICS
# -----------------------------------------------------------------------------

print(f"Total triples: {triples_count}")
print(f"Entity types ({len(entities)}):")
for entity_type, entity_list in entities.items():
    unique_count = len(set(entity_list))
    print(f"  - {entity_type}: {unique_count}")
    if entity_type == 'uncertainty':
        print(f"    Unique uncertainty values: {set(entity_list)}")

print(f"Relations ({len(relations)}):")
for rel, count in relations.items():
    print(f"  - {rel}: {count}")


# -----------------------------------------------------------------------------
# CORRELATION ANALYSIS
# -----------------------------------------------------------------------------

def from_cat_to_int(list_cat: list, reset: int = 0) -> list[int]:
    """
    Convert a list of categorical values to integer codes.

    Args:
        list_cat (list): Categorical values.
        reset (int): Starting integer code (useful to offset mappings between features).

    Returns:
        list[int]: Integer-encoded representation of the categorical list.
    """
    mapping = {}
    index = reset
    result = []
    for e in list_cat:
        if e not in mapping:
            mapping[e] = index
            index += 1
        result.append(mapping[e])
    return result


# Load dataset for correlation analysis
df = pd.read_csv(df_path)

# Standardize column names
df = df.rename(columns={
    'author': 'Author',
    'experiment_type': 'Experiment Type',
    'reactor': 'Reactor',
    'similarity': 'Similarity',
    'target': 'Target',
    'uncertainty': 'Uncertainty',
    'year': 'Year'
})

# Identify categorical vs numerical columns
cat_cols = set(df.select_dtypes(include=[object]).columns)
num_cols = set(df.columns) - cat_cols
col_types = {**{c: 'cat' for c in cat_cols}, **{c: 'num' for c in num_cols}}

# Initialize correlation matrix
corr_df = pd.DataFrame(columns=sorted(col_types.keys()), index=sorted(col_types.keys()))

# Compute pairwise correlations
for x, y in itertools.combinations(col_types.keys(), 2):
    if col_types[x] == 'cat' and col_types[y] == 'cat':
        # Chi-square test for categorical pairs
        data = np.array([
            from_cat_to_int(df[x]),
            from_cat_to_int(df[y], reset=len(df[x]))
        ])
        X2 = chi2_contingency(data, correction=False)[0]
        n = np.sum(data)
        minDim = min(data.shape) - 1
        c = np.sqrt((X2 / n) / minDim)
        p = 0
    elif col_types[x] == 'num' and col_types[y] == 'num':
        # Pearson correlation
        c, p = pearsonr(df[x], df[y])
    elif col_types[x] == 'cat' and col_types[y] == 'num':
        # Point-biserial correlation (categorical vs numeric)
        c, p = pointbiserialr(from_cat_to_int(df[x]), df[y])
    elif col_types[x] == 'num' and col_types[y] == 'cat':
        c, p = pointbiserialr(from_cat_to_int(df[y]), df[x])

    corr_df.loc[x, y] = c
    corr_df.loc[y, x] = c

# Fill diagonal with perfect correlation
np.fill_diagonal(corr_df.values, 1)

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------

# Prepare correlation matrix for plotting
corr_df = corr_df.astype('float64')
mask = np.triu(np.ones_like(corr_df, dtype=bool))

# Create figure and heatmap
plt.figure(figsize=(11, 9))
sns.heatmap(
    corr_df,
    mask=mask,
    cmap=sns.diverging_palette(230, 20, as_cmap=True),
    vmax=1,
    vmin=-1,
    center=0,
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .5},
    annot=True,
    annot_kws={"size": 14}
)

plt.title("Correlation Heatmap Across Dataset Features", fontsize=16)
plt.tight_layout()
plt.show()
