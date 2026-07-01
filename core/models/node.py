# core/models/node.py
import hashlib
from typing import Optional, Dict, Any, List, ClassVar
from enum import Enum
# FIX: Added BaseModel and swapped field_validator for model_validator
from pydantic import BaseModel, Field, model_validator
from core.models.base import SpatialRAGBase

# ---- MODALITY CATEGORIES ----
class ModalityCategory(str, Enum):
    DOCUMENT_STRUCTURE = "document_structure"
    TEXTUAL_CONTENT = "textual_content"
    METADATA = "metadata"
    TABLE_CONTAINER = "table_container"
    TABLE_STRUCTURE = "table_structure"
    TABLE_CONTENT = "table_content"
    CHART_CONTAINER = "chart_container"
    CHART_STRUCTURE = "chart_structure"
    CHART_DATA = "chart_data"
    DIAGRAM_CONTAINER = "diagram_container"
    DIAGRAM_COMPONENT = "diagram_component"
    DIAGRAM_CONNECTION = "diagram_connection"
    DIAGRAM_LABEL = "diagram_label"
    FLOWCHART_CONTAINER = "flowchart_container"
    FLOWCHART_NODE = "flowchart_node"
    FLOWCHART_EDGE = "flowchart_edge"
    FORMULA = "formula"
    EQUATION = "equation"
    IMAGE = "image"
    SHAPE = "shape"
    VIDEO_FRAME = "video_frame"
    AUDIO_SEGMENT = "audio_segment"
    TIMELINE_MARKER = "timeline_marker"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    UNKNOWN = "unknown"


class Granularity(str, Enum):
    ROOT = "root"
    SECTION = "section"
    PAGE = "page"
    BLOCK = "block"
    ELEMENT = "element"
    TOKEN = "token"


class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    page: int = 0
    confidence: float = 1.0
    dpi: Optional[float] = None


class Node(SpatialRAGBase):
    MAX_CONTENT_LENGTH: ClassVar[int] = 8192

    doc_id: str
    parent_id: Optional[str] = None
    subgraph_id: Optional[str] = None
    subgraph_role: str = "member"  # "container" | "member"

    modality: str = "unknown"
    modality_category: ModalityCategory = ModalityCategory.UNKNOWN

    granularity: Granularity = Granularity.ELEMENT
    sequence_index: int = 0

    content: str = ""
    content_hash: str = ""
    content_truncated: bool = False

    bbox: Optional[BoundingBox] = None
    image_path: Optional[str] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None

    embedding_refs: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of model_name to Qdrant point ID"
    )

    processor_name: str
    processor_version: str

    # FIX: Changed to model_validator so we can mutate the whole instance directly
    @model_validator(mode='after')
    def validate_and_hash_content(self) -> 'Node':
        current_content = self.content
        if len(current_content) > self.MAX_CONTENT_LENGTH:
            truncated = current_content[:self.MAX_CONTENT_LENGTH]
            new_hash = hashlib.sha256(truncated.encode()).hexdigest()
            
            # Bypass Pydantic's __setattr__ hook to prevent infinite recursion
            self.__dict__['content'] = truncated
            self.__dict__['content_truncated'] = True
            self.__dict__['content_hash'] = new_hash
        else:
            new_hash = hashlib.sha256(current_content.encode()).hexdigest()
            self.__dict__['content_truncated'] = False
            self.__dict__['content_hash'] = new_hash
            
        return self