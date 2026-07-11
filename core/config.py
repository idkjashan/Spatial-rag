# core/config.py
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from core.models.processor import ProcessorCapability
from pydantic_settings import BaseSettings, SettingsConfigDict

class ModelRegistry(BaseModel):
    embedders: Dict[str, str] = Field(default_factory=dict)
    llms: Dict[str, str] = Field(default_factory=dict)
    vlms: Dict[str, str] = Field(default_factory=dict)
    rerankers: Dict[str, str] = Field(default_factory=dict)

class PluginRegistry(BaseModel):
    active_parsers: List[ProcessorCapability] = Field(default_factory=list)
    active_embedders: List[str] = Field(default_factory=list)

class DatabaseConfig(BaseSettings):
    postgres_dsn: str = Field(
        default="postgresql://spatialrag:localdev@postgres:5432/spatialrag",
        validation_alias="META_DB_URL"
    )
    
    neo4j_url: str = Field(
        default="bolt://neo4j:7687",
        validation_alias="GRAPH_DB_URL"
    )
    neo4j_user: str = Field(default="neo4j", validation_alias="GRAPH_DB_USER")
    neo4j_password: str = Field(default="password", validation_alias="GRAPH_DB_PASSWORD")
    
    qdrant_url: str = Field(
        default="http://qdrant:6333",
        validation_alias="VECTOR_DB_URL"
    )
    qdrant_api_key: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",             # Fallback to local .env file if running locally
        env_file_encoding="utf-8",
        extra="ignore",              # Ignores other random system env vars so they don't cause errors
        populate_by_name=True        # Allows you to still instantiate the class using standard kwargs
    )

class EmbeddingStrategy(BaseModel):
    name: str
    vector_name: str      # named vector in Qdrant
    dimension: int
    model_name: str       # e.g., "BGE-M3", "CLIP"
    type: str = "dense"   # "dense", "sparse", "multi-vector"

class EngineConfig(BaseModel):
    tenant_id: str = "default"
    max_content_length: int = 8192
    image_cache_path: str = "/tmp/spatialrag_images"

    db: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # Embedding strategies to generate for each node/edge
    node_embedding_strategies: List[EmbeddingStrategy] = Field(
        default_factory=lambda: [
            EmbeddingStrategy(name="bge-m3-dense", vector_name="dense_text", dimension=1024, model_name="BGE-M3", type="dense"),
            EmbeddingStrategy(name="clip", vector_name="visual_clip", dimension=512, model_name="CLIP", type="dense"),
        ]
    )
    edge_embedding_strategies: List[EmbeddingStrategy] = Field(
        default_factory=lambda: [
            EmbeddingStrategy(name="bge-m3-evidence", vector_name="evidence_dense", dimension=1024, model_name="BGE-M3", type="dense"),
        ]
    )
    active_node_strategies: List[str] = Field(default_factory=lambda: ["bge-m3-dense"])
    active_edge_strategies: List[str] = Field(default_factory=lambda: ["bge-m3-evidence"])