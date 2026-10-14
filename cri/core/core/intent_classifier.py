
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("NL2SQL")


# ─────────────────────────────────────────────────────────────────────────────
# Intent descriptor sentences
# ─────────────────────────────────────────────────────────────────────────────
# Each intent gets 10 semantically diverse paraphrases.
# Rule: describe WHAT THE SQL WILL LOOK LIKE, not just phrasing style.
# That grounds the embedding in SQL semantics, not surface words.
# You can add more sentences to any intent to improve its coverage.
# ─────────────────────────────────────────────────────────────────────────────

INTENT_DESCRIPTORS: Dict[str, List[str]] = {

    "aggregation": [
        "count the total number of records",
        "how many items exist in the system",
        "calculate the sum of a numeric field",
        "what is the total quantity produced",
        "find the average value across all rows",
        "compute the minimum or maximum of a column",
        "how much stock is currently available",
        "aggregate a numeric field across all rows using COUNT or SUM",
        "what is the overall production count",
        "total output across all work orders",
    ],

    "group_by": [
        "count records grouped by a category",
        "break down totals by status",
        "summarise quantity per work order",
        "how many items for each type",
        "distribution of records by department",
        "aggregate values for each distinct category",
        "total quantity per material grouped",
        "count per status using GROUP BY",
        "group the results by a dimension and show totals",
        "pivot results by a categorical column",
    ],

    "top_bottom": [
        "show the top 10 records by quantity",
        "find the highest value items in the table",
        "list the bottom 5 entries by cost",
        "which work orders have the most output",
        "rank materials by total usage",
        "get the lowest performing machines",
        "top N results ordered by a numeric metric",
        "best or worst records by a measure",
        "most frequently issued materials",
        "least used resources ranked by count",
    ],

    "time_based": [
        "records created this week",
        "filter rows by a date range",
        "work orders from the last 7 days",
        "transactions completed in the current month",
        "activity logged since a specific date",
        "show data entered today",
        "recent entries within a time window",
        "filter by creation date or modification date",
        "last 30 days of production entries",
        "entries between two calendar dates",
    ],

    "existence": [
        "check if a specific record exists",
        "is there any open work order for this machine",
        "does this material exist in the system",
        "verify whether a particular entry is present",
        "confirm if a condition is true or false",
        "is at least one record matching these criteria",
        "confirm presence or absence of a record",
        "yes or no question about whether data exists",
        "is there available stock for this item",
        "are there any pending or overdue items",
    ],

    "search": [
        "find records whose name contains a keyword",
        "search for items matching a text pattern",
        "work orders whose name starts with a given prefix",
        "partial text match on a name or description column",
        "filter using a wildcard or LIKE pattern",
        "items whose description includes a specific word",
        "records ending with a particular suffix",
        "full or partial text search across a column",
        "find entries by partial name match",
        "locate rows containing a search string",
    ],

    "lookup": [
        "what is the status of a specific work order",
        "get the BOM name for a particular item",
        "retrieve the value of one field for a known record",
        "find the details of a specific entity by its ID",
        "show the quantity for a particular work order number",
        "look up the due date of a specific order",
        "what is the plant assigned to this material code",
        "fetch a single attribute of a known record by key",
        "get the description for a specific code or ID",
        "retrieve one column value given a primary key",
    ],

    "conditional": [
        "show work orders where the status is released",
        "filter records by a specific field value",
        "list items that meet multiple criteria",
        "work orders with plant equal to a specific location",
        "records where quantity is greater than a threshold",
        "find all entries with a particular flag set",
        "filter by multiple WHERE conditions",
        "show only active or enabled records",
        "entries where a column equals a given value",
        "display rows matching a set of filter criteria",
    ],

    "sort_by": [
        "list all work orders ordered by creation date",
        "show all materials sorted by quantity descending",
        "display all records in alphabetical order by name",
        "order the full result set by a column",
        "sort all entries from newest to oldest",
        "list everything in ascending order by name",
        "show all records sorted by priority without a row limit",
        "retrieve all entries ordered by a specific field",
        "order all results by date descending",
        "sorted listing of the complete result set",
    ],

    "simple": [
        "list all work orders in the table",
        "show all materials without any filter",
        "get all records from the table",
        "display everything stored in this table",
        "fetch all rows with no condition",
        "show me the full unfiltered list",
        "retrieve all entries from the database",
        "what are all the available work orders",
        "list all records in the system",
        "show every row in the table",
    ],

    "relational_lookup": [
        "get work order details along with its BOM information from another table",
        "show the material name for each work order by joining tables",
        "fetch order details joined with plant information from a related table",
        "retrieve data that spans two or more tables using a join",
        "look up an attribute from a related or linked table",
        "find the supplier name for each purchase order using a join",
        "join work order table with material master to get description",
        "show combined information from two related tables",
        "get the BOM details for a specific work order by joining",
        "retrieve linked records by joining across tables",
    ],

    "relational_aggregation": [
        "total quantity issued per work order joined with material details",
        "count of work orders grouped by BOM using a join",
        "sum of production across work orders joined with the plant table",
        "aggregate data that requires joining multiple tables together",
        "total output per machine joined with the work order table",
        "count transactions grouped by material name from a related table",
        "sum of quantity per order joined with the product master table",
        "aggregation across joined tables using GROUP BY",
        "calculate grouped totals that combine data from two related tables",
        "sum or count with GROUP BY over a multi-table join",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Zero-shot intent classifier backed by sentence embeddings.

    Instantiate ONCE at app startup.  classify() is pure numpy at query time.
    """

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self._labels: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._build_centroids()

    def _build_centroids(self) -> None:
        """Embed descriptor sentences and average into one centroid per intent."""
        log.info("[intent_classifier] Building intent centroids …")
        centroids: Dict[str, np.ndarray] = {}

        for intent, sentences in INTENT_DESCRIPTORS.items():
            vecs = self._embedder.encode(sentences, normalize_embeddings=True)
            centroid = vecs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[intent] = centroid / norm if norm > 0 else centroid

        self._labels = list(centroids.keys())
        self._matrix = np.stack([centroids[l] for l in self._labels])  # (n, dim)
        log.info(
            "[intent_classifier] Ready — %d intents, embedding dim=%d",
            len(self._labels), self._matrix.shape[1],
        )

    def classify(self, question: str) -> str:
        """Return the single best-matching intent label for the question."""
        q_vec = self._embedder.encode(question, normalize_embeddings=True)
        scores: np.ndarray = self._matrix @ q_vec          # cosine sim per intent
        ranked = np.argsort(scores)[::-1]

        best  = self._labels[ranked[0]]
        s1    = float(scores[ranked[0]])
        s2    = float(scores[ranked[1]])
        log.info(
            "[intent_classifier] %r → %r (%.3f)  runner-up=%r (%.3f)",
            question[:60], best, s1, self._labels[ranked[1]], s2,
        )
        return best

    def scores(self, question: str) -> Dict[str, float]:
        """Return similarity score for every intent — use this for debugging."""
        q_vec = self._embedder.encode(question, normalize_embeddings=True)
        raw: np.ndarray = self._matrix @ q_vec
        return dict(sorted(
            {label: float(raw[i]) for i, label in enumerate(self._labels)}.items(),
            key=lambda x: x[1], reverse=True,
        ))
