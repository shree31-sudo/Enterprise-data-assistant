

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

log = logging.getLogger("NL2SQL")

@dataclass
class TableRelationship:
    """Represents a relationship between two tables."""
    left_table: str
    right_table: str
    cardinality: str  # "one_to_one", "one_to_many", "many_to_many"
    nullable: bool = False  # True if the relationship is optional
    fk_constraint: Optional[str] = None
    left_columns: List[str] = field(default_factory=list)
    right_columns: List[str] = field(default_factory=list)
    join_type: Optional[str] = None  # Explicit join type override
    
    def get_join_type(self, context: str = "default") -> str:
        """
        Determine the appropriate JOIN type based on relationship metadata.
        Returns "INNER" or "LEFT".
        """
        # If an explicit join_type is specified, use it
        if self.join_type:
            return self.join_type
        
        # If nullable is True, the relationship is optional - use LEFT JOIN
        if self.nullable:
            return "LEFT"
        
        # For one_to_many, check the context
        if self.cardinality == "one_to_many":
            # If we're going from parent to child and the parent is the "one" side
            if context == "parent_to_child":
                # If the child table is optional (nullable FK), use LEFT JOIN
                if self.nullable:
                    return "LEFT"
                # Otherwise INNER JOIN is fine
                return "INNER"
            else:
                # Going from child to parent - always INNER JOIN
                return "INNER"
        
        # Default to INNER JOIN
        return "INNER"


class RelationshipResolver:
    """Resolves table relationships and determines appropriate JOIN types."""
    
    def __init__(self, relationship_data: Dict = None):
        self._relationships: Dict[Tuple[str, str], TableRelationship] = {}
        self._table_graph: Dict[str, Set[str]] = {}
        self._nullable_fks: Set[Tuple[str, str]] = set()  # (table, column) pairs
        
        if relationship_data:
            self._build_index(relationship_data)
        else:
            log.warning("[relationship] No relationship data provided")
    
    def _build_index(self, relationship_data: Dict):
        """Build indexes from relationship metadata."""
        for key, data in relationship_data.get("table_pairs", {}).items():
            # Parse the key (could be "TABLE1-TABLE2" or tuple)
            if isinstance(key, str):
                parts = key.split("-")
                if len(parts) == 2:
                    left, right = parts[0], parts[1]
                else:
                    log.warning("[relationship] Invalid key format: %s", key)
                    continue
            else:
                left, right = key
            
            # Store both directions for easy lookup
            rel = TableRelationship(
                left_table=left,
                right_table=right,
                cardinality=data.get("cardinality", "one_to_many"),
                nullable=data.get("nullable", False),
                fk_constraint=data.get("fk_constraint"),
                left_columns=data.get("left_columns", []),
                right_columns=data.get("right_columns", []),
                join_type=data.get("join_type"),
            )
            
            # Store in both directions
            self._relationships[(left.upper(), right.upper())] = rel
            self._relationships[(right.upper(), left.upper())] = rel
            
            # Build graph
            self._table_graph.setdefault(left.upper(), set()).add(right.upper())
            self._table_graph.setdefault(right.upper(), set()).add(left.upper())
            
            # Track nullable FKs
            if rel.nullable and rel.right_columns:
                for col in rel.right_columns:
                    self._nullable_fks.add((right.upper(), col.upper()))
        
        log.info(
            "[relationship] Built index: %d relationships, %d tables, %d nullable FKs",
            len(self._relationships) // 2,  # Divide by 2 because we stored both directions
            len(self._table_graph),
            len(self._nullable_fks)
        )
    
    def get_relationship(self, table1: str, table2: str) -> Optional[TableRelationship]:
        """Get relationship between two tables."""
        key = (table1.upper(), table2.upper())
        return self._relationships.get(key)
    
    def is_nullable_fk(self, table: str, column: str) -> bool:
        """Check if a column is a nullable foreign key."""
        return (table.upper(), column.upper()) in self._nullable_fks
    
    def get_join_type_for_step(
        self,
        from_table: str,
        to_table: str,
        direction: str = "forward"
    ) -> str:
        """
        Get the join type for a single join step.
        
        Args:
            from_table: The source table
            to_table: The target table
            direction: "forward" (from source to target) or "reverse"
        
        Returns:
            "INNER" or "LEFT"
        """
        rel = self.get_relationship(from_table, to_table)
        if rel is None:
            # If no relationship found, check if the FK is nullable
            # This is a fallback for when relationship metadata is missing
            log.debug(
                "[relationship] No relationship for %s <-> %s, checking nullable FK",
                from_table, to_table
            )
            # Look for pattern: to_table has a column that matches from_table's PK
            from_pk = self._get_pk_columns(from_table)
            for col in from_pk:
                if self.is_nullable_fk(to_table, col):
                    return "LEFT"
            return "INNER"
        
        # Use the relationship's join type
        return rel.get_join_type(direction)
    
    def _get_pk_columns(self, table: str) -> Set[str]:
        """Get primary key columns for a table from the schema cache."""
        from main import FULL_SCHEMA_CACHE, META
        
        pk = set()
        # Check FULL_SCHEMA_CACHE first
        if table in FULL_SCHEMA_CACHE:
            for col in FULL_SCHEMA_CACHE[table].get("pk", []):
                pk.add(col.upper())
        
        # Fallback to META
        if table in META:
            for col, info in META[table].get("columns", {}).items():
                if isinstance(info, dict) and info.get("is_pk", False):
                    pk.add(col.upper())
        
        return pk
    
    def get_join_type_for_path(
        self,
        path: List[Dict],
        base_table: str,
        all_tables: Set[str]
    ) -> str:
        """
        Determine the join type for each step in a join path.
        
        Returns:
            The appropriate JOIN type for the entire path
        """
        if not path:
            return "INNER"
        
        # Determine the overall join type based on the most permissive requirement
        join_types = []
        
        for step in path:
            from_table = step["from_table"]
            to_table = step["to_table"]
            
            # Determine the direction
            direction = "forward"
            
            # Get the join type for this step
            join_type = self.get_join_type_for_step(from_table, to_table, direction)
            join_types.append(join_type)
        
        # If any step requires LEFT JOIN, use LEFT JOIN for the whole path
        # because the first LEFT JOIN will preserve all rows from the left side
        if "LEFT" in join_types:
            return "LEFT"
        
        return "INNER"