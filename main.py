# main.py
from fastapi import FastAPI
from core.config import EngineConfig
from core.models.node import Node, ModalityCategory, Granularity

app = FastAPI(title="SpatialRAG Core Orchestrator")
config = EngineConfig()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "SpatialRAG Phase 0 Core Engine",
        "tenant_context": config.tenant_id
    }

@app.get("/health/verification")
def health_verification():
    # Sanity check testing internal validation logic
    try:
        test_node = Node(
            id="00000000-0000-0000-0000-000000000000",
            doc_id="11111111-1111-1111-1111-111111111111",
            processor_name="health_check",
            processor_version="1.0.0",
            modality_category=ModalityCategory.TEXTUAL_CONTENT,
            granularity=Granularity.BLOCK,
            content="Docker integration test verified."
        )
        return {"engine_status": "healthy", "model_integrity": "passed", "node_hash": test_node.content_hash}
    except Exception as e:
        return {"engine_status": "degraded", "error": str(e)}