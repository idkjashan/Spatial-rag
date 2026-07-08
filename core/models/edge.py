# core/models/edge.py
from typing import Optional, Dict, Any
from pydantic import Field
from core.models.base import SpatialRAGBase
from enum import Enum

# ---- EDGE CATEGORIES ----
class EdgeCategory(str, Enum):
    HIERARCHY = "hierarchy"
    READ_ORDER = "read_order"
    LAYOUT = "layout"
    CAPTION = "caption"
    REFERENCE = "reference"
    DEFINITION = "definition"
    LIST_HIERARCHY = "list_hierarchy"
    TABLE_HIERARCHY = "table_hierarchy"
    TABLE_SEMANTICS = "table_semantics"
    TABLE_AGGREGATION = "table_aggregation"
    PHYSICAL_CONNECTION = "physical_connection"
    FLOW_DIRECTION = "flow_direction"
    SIGNAL_FLOW = "signal_flow"
    ENERGY_FLOW = "energy_flow"
    CHART_HIERARCHY = "chart_hierarchy"
    CHART_SEMANTICS = "chart_semantics"
    CHART_GROUPING = "chart_grouping"
    FLOW_LOGIC = "flow_logic"
    LOOP_STRUCTURE = "loop_structure"
    TIME_ORDER = "time_order"
    TIME_CONTAINMENT = "time_containment"
    SPATIAL_RELATION = "spatial_relation"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    SEMANTIC_RELATION = "semantic_relation"
    UNKNOWN = "unknown"


class Edge(SpatialRAGBase):
    source_id: str
    target_id: str

    type: str = "unknown"
    type_category: EdgeCategory = EdgeCategory.UNKNOWN

    is_bidirectional: bool = False

    weight: float = 1.0
    confidence: float = 1.0

    creator_processor: str = "core"
    creator_version: str = "1.0.0"

    edge_meta: Dict[str, Any] = Field(default_factory=dict)
    evidence: Optional[str] = None

    embedding_refs: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of model_name to Qdrant point ID for the edge's evidence text"
    )