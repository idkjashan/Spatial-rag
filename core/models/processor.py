# core/models/processor.py
from pydantic import Field, ConfigDict
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
from core.models.base import SpatialRAGBase

class ProcessorCapability(str, Enum):
    TEXT_PARSER = "text_parser"
    TABLE_PARSER = "table_parser"
    DIAGRAM_PARSER = "diagram_parser"
    CHART_PARSER = "chart_parser"
    FLOWCHART_PARSER = "flowchart_parser"
    VLM_PARSER = "vlm_parser"
    FORMULA_PARSER = "formula_parser"
    EMBEDDER = "embedder"
    RERANKER = "reranker"

class ProcessorManifest(SpatialRAGBase):
    model_config = ConfigDict(protected_namespaces=())
    processor_name: str
    version: str
    capability: ProcessorCapability
    model_parameters: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    loaded_memory_mb: Optional[float] = None

    inputs_required: List[str] = Field(default_factory=list)
    outputs_produced: List[str] = Field(default_factory=list)