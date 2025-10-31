"""
BaseClasses.py
--------------

Defines foundational data structures for representing synthetic experimental
data and converting them into knowledge graph triples.

Classes:
    - Triple: Represents a single (head, relation, tail) triple.
    - Experiment: Represents a simulated experimental observation, capable of
      generating its own attributes, computing similarity relationships, and
      exporting itself as knowledge graph triples or structured records.

Author: Edoardo Ramalli
"""

import os
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional


class Triple:
    """
    Represents a knowledge graph triple (head, relation, tail).

    Attributes:
        head (str): Subject of the triple.
        relation (str): Predicate describing the relation.
        tail (str): Object of the triple.
    """

    def __init__(self, head: str, relation: str, tail: str) -> None:
        self.head = str(head)
        self.relation = str(relation)
        self.tail = str(tail)

    def to_string(self, sep: str = '\t', order: Tuple[str, str, str] = ('head', 'relation', 'tail')) -> str:
        """
        Convert the triple into a delimited string.

        Args:
            sep (str): Separator between triple elements. Defaults to tab ('\t').
            order (tuple): Defines order of elements ('head', 'relation', 'tail').

        Returns:
            str: Triple formatted as a string (e.g., "head\trelation\ttail").
        """
        return sep.join(str(self.__dict__[i]) for i in order)


class Experiment:
    """
    Represents a simulated experimental dataset entry with randomizable attributes.

    Each Experiment instance defines relationships (triples) connecting its attributes
    (type, reactor, author, etc.) and can determine similarity or "closeness" to other experiments.

    Attributes:
        exp_id (str): Unique experiment identifier.
        experiment_type (str): Type of experiment (e.g., ignition delay measurement).
        reactor (str): Reactor used in the experiment.
        author (str): Author assigned to the experiment.
        year (int): Year in which the experiment took place.
        target (str): Target species or observable.
        similarity (float): Similarity value between 0 and 1.
        uncertainty (Optional[float]): Measurement uncertainty (if modeled).
        close (Set[str]): Set of other experiment IDs considered "close" to this one.
    """

    # --- Class-level constants and lookup tables ---
    possible_experiment_types: Dict[str, List[str]] = {
        'ignition delay measurement': ['shock tube', 'rapid compression machine'],
        'laminar burning velocity measurement': ['flame'],
        'outlet concentration measurement': ['shock tube', 'stirred reactor', 'flow reactor'],
        'concentration time profile measurement': ['shock tube', 'stirred reactor', 'flow reactor'],
        'jet stirred reactor measurement': ['shock tube', 'stirred reactor', 'flow reactor'],
        'burner stabilized flame speciation measurement': ['flame'],
    }

    possible_year_of_activity: List[Tuple[int, int]] = [(i, i + 10) for i in range(1950, 2030, 10)]
    list_uncertainties: List[float] = [round(x, 3) for x in np.linspace(0, 1, 11)]
    list_similarities: List[float] = [round(x, 3) for x in np.linspace(0, 1, 11)]
    possible_target: List[str] = [
        'H2', 'IDT', 'C0', 'H20', 'CO2', 'NH3', 'CH4', 'CH3', 'C5H6', 'C6H6', 'H2S', 'C7H8'
    ]
    n_possible_author: int = 50

    # Author metadata and uncertainty placeholders
    authors_activity: Dict[str, Tuple[int, int]] = {}
    possible_uncertainty_author: Dict[str, Tuple[float, float]] = {}
    possible_uncertainty_author_year: Dict = {}
    possible_uncertainty_author_year_exp_type: Dict = {}
    possible_uncertainty_author_year_exp_type_target: Dict = {}

    # Initialize author activity
    for i in range(n_possible_author):
        c_author = f'Author {i}'
        authors_activity[c_author] = random.choice(possible_year_of_activity)

    # --- Sampling helper methods ---
    def pick_experiment_type(self) -> str:
        """Randomly choose an experiment type."""
        return random.choice(list(self.possible_experiment_types.keys()))

    def pick_reactor(self) -> str:
        """Randomly choose a valid reactor based on the current experiment type."""
        return random.choice(self.possible_experiment_types[self.experiment_type])

    def pick_author(self) -> str:
        """Randomly choose an author."""
        return random.choice(list(self.authors_activity.keys()))

    def pick_year(self) -> int:
        """Pick a year within the author’s activity window."""
        active_years = self.authors_activity[self.author]
        return random.randint(active_years[0], active_years[1])

    def pick_target(self) -> str:
        """Randomly choose a chemical or target observable."""
        return random.choice(self.possible_target)

    def pick_similarity(self) -> float:
        """Randomly choose a similarity value."""
        return random.choice(self.list_similarities)

    def pick_uncertainty(self) -> Optional[float]:
        """
        Pick an uncertainty value.

        Note:
            Currently disabled (returns None), but can be reactivated by
            uncommenting one of the pre-defined "Case" models below.
        """
        return None

    # --- Initialization ---
    def __init__(self, exp_id: int) -> None:
        """
        Initialize an Experiment with random attributes.

        Args:
            exp_id (int): Numeric experiment ID.
        """
        self.exp_id = f'ID:{exp_id}'
        self.experiment_type = self.pick_experiment_type()
        self.reactor = self.pick_reactor()
        self.author = self.pick_author()
        self.year = self.pick_year()
        self.target = self.pick_target()
        self.similarity = self.pick_similarity()
        self.uncertainty = self.pick_uncertainty()
        self.close: Set[str] = set()

    # --- Similarity and Closeness ---
    def meta_similar(self, other_exp: 'Experiment') -> bool:
        """
        Determine if another experiment is structurally similar.

        Experiments are considered 'meta-similar' if they share the same
        type, reactor, and target.

        Args:
            other_exp (Experiment): Experiment to compare.

        Returns:
            bool: True if meta-similar, otherwise False.
        """
        return (
            self.experiment_type == other_exp.experiment_type
            and self.reactor == other_exp.reactor
            and self.target == other_exp.target
        )

    def similar(self, other_exp: 'Experiment') -> bool:
        """
        Determine if another experiment is similar in both structure and similarity score.

        Args:
            other_exp (Experiment): Experiment to compare.

        Returns:
            bool: True if both meta-similar and |similarity difference| ≤ 0.15.
        """
        return self.meta_similar(other_exp) and abs(self.similarity - other_exp.similarity) <= 0.15

    # --- Representation and Export ---
    def __repr__(self) -> str:
        """Readable summary of the experiment (useful for debugging)."""
        return (
            f"<{self.exp_id} -> {self.experiment_type[:8]} | {self.reactor} | "
            f"{self.author} | {self.year} | {self.target} | "
            f"{self.similarity} | {self.uncertainty}>"
        )

    def to_kg(self, sep: str = '\t', order: Tuple[str, str, str] = ('head', 'relation', 'tail')) -> List[str]:
        """
        Convert the experiment into a list of knowledge graph triples.

        Args:
            sep (str): Separator for triple string formatting.
            order (tuple): Order of triple elements.

        Returns:
            list[str]: List of triples in string form.
        """
        triples = [
            Triple(self.exp_id, 'of_type', f'TY_{self.experiment_type}').to_string(sep, order),
            Triple(self.exp_id, 'of_reactor', f'R_{self.reactor}').to_string(sep, order),
            Triple(self.exp_id, 'has_author', self.author).to_string(sep, order),
            Triple(self.exp_id, 'of_year', f'Y_{self.year}').to_string(sep, order),
            Triple(self.exp_id, 'of_target', f'T_{self.target}').to_string(sep, order),
            Triple(self.exp_id, 'has_similarity', f'S_{self.similarity}').to_string(sep, order),
            Triple(self.exp_id, 'has_uncertainty', f'U_{self.uncertainty}').to_string(sep, order),
        ]

        # Add closeness relations if any
        for close_id in self.close:
            triples.append(Triple(self.exp_id, 'is_close', close_id).to_string(sep, order))

        return triples

    def to_record(self) -> Dict[str, object]:
        """
        Convert experiment attributes into a flat dictionary record.

        Returns:
            dict: Record with experiment metadata.
        """
        return {
            'experiment_type': self.experiment_type,
            'reactor': self.reactor,
            'author': self.author,
            'year': self.year,
            'target': self.target,
            'similarity': self.similarity,
            'uncertainty': self.uncertainty,
        }
