# core/config.py
from typing import Dict, List
from pydantic import BaseModel, Field
from core.models.processor import ProcessorCapability

class ModelRegistry(BaseModel):
    embedders: Dict[str, str] = Field(default_factory=dict)
    llms: Dict[str, str] = Field(default_factory=dict)
    vlms: Dict[str, str] = Field(default_factory=dict)
    rerankers: Dict[str, str] = Field(default_factory=dict)

class PluginRegistry(BaseModel):
    active_parsers: List[ProcessorCapability] = Field(default_factory=list)
    active_embedders: List[str] = Field(default_factory=list)

class EngineConfig(BaseModel):
    tenant_id: str = "default"
    max_content_length: int = 8192
    graph_db_url: str = "bolt://neo4j:7687"
    vector_db_url: str = "http://qdrant:6333"
    meta_db_url: str = "postgresql://spatialrag:localdev@postgres:5432/spatialrag"
    image_cache_path: str = "/tmp/spatialrag_images"

    models: ModelRegistry = Field(default_factory=ModelRegistry)
    plugins: PluginRegistry = Field(default_factory=PluginRegistry)