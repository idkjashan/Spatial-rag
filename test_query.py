import os
import sys
import json
import asyncio
import warnings
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Suppress Pydantic V2 namespace warnings
warnings.filterwarnings(
    "ignore",
    message='Field "model_name" has conflict with protected namespace "model_".',
    category=UserWarning,
    module="pydantic"
)

load_dotenv()

from core.config import EngineConfig
from core.query.router import QueryRouter, RetrievalPlan
from core.query.retriever import GraphRetriever
from core.query.hydrator import ContextHydrator
from core.stores.postgres_store import PostgresStore
from qdrant_client import AsyncQdrantClient
from openai import AsyncOpenAI

try:
    from rich import print
    from rich.panel import Panel
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

# ==========================================
# Phase 4.3: LLM Synthesis Implementation
# ==========================================

async def run_synthesis_phase(router_client: QueryRouter, final_context_string: str, original_query: str):
    """
    Phase 4.3 (LLM Synthesis Engine).
    Passes the context string to the generative SLM to construct the final answer.
    """
    if RICH_AVAILABLE:
        print(f"\n[bold yellow]🤖 Phase 4.3: Generating Answer via LLM Synthesis...[/bold yellow]")
    else:
        print(f"\n🤖 Phase 4.3: Generating Answer via LLM Synthesis...")

    system_prompt = (
        "You are a precise engineering assistant analyzing data extracted from a Graph RAG system.\n"
        "Your task is to answer the user query using ONLY the provided context below. "
        "If the context does not contain enough evidence to answer, explicitly state that.\n\n"
        "--- START CONTEXT SEED ---\n"
        f"{final_context_string}\n"
        "--- END CONTEXT SEED ---"
    )

    user_prompt = f"User Query: {original_query}"

    try:
        # Re-use the decoupled client credentials stored inside the initialized router
        response = await router_client.llm_client.chat.completions.create(
            model=router_client.slm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0  # Force exact factual reproduction
        )
        
        answer = response.choices[0].message.content
        
        if RICH_AVAILABLE:
            console.print(Panel(answer, title="🎯 RAG System Synthesized Answer", border_style="bold green"))
        else:
            print("\n🎯 RAG System Synthesized Answer:")
            print(answer)
            print("==========================================")
            
    except Exception as e:
        print(f"❌ Failed to synthesize answer via generative API: {e}")

# ==========================================
# Component Initializers & Monkeypatching
# ==========================================

async def init_router(config: EngineConfig) -> QueryRouter:
    print("🔌 Connecting Router clients...")
    qdrant = AsyncQdrantClient(url=config.db.qdrant_url, api_key=config.db.qdrant_api_key)
    
    return QueryRouter(
        qdrant_client=qdrant,
        qdrant_node_collection="nodes",
        slm_model=os.getenv("SLM_MODEL", "openai/gpt-oss-20b"), 
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        llm_api_key=os.getenv("GROQ_API_KEY"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "jina-embeddings-v4"),
        embed_base_url=os.getenv("EMBED_BASE_URL", "https://api.jina.ai/v1"),
        embed_api_key=os.getenv("JINA_API_KEY")
    )

async def init_retriever(config: EngineConfig) -> GraphRetriever:
    print("🔌 Connecting GraphRetriever to Neo4j and Qdrant...")
    return GraphRetriever(
        db_config=config.db,
        qdrant_node_collection="nodes",
        qdrant_edge_collection="edges",
        embed_base_url=os.getenv("EMBED_BASE_URL", "https://api.jina.ai/v1"),
        embed_api_key=os.getenv("JINA_API_KEY"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "jina-embeddings-v4"),
        max_final_nodes=10,
        min_expand_threshold=10
    )

def patch_context_hydrator():
    """
    Monkeypatches ContextHydrator._fetch_nodes to dynamically inject 
    missing required Pydantic properties, skipping instantiation drops.
    """
    from psycopg2.extras import RealDictCursor
    from core.models.node import Node

    async def patched_fetch_nodes(self, node_ids: list) -> list:
        if not node_ids: return []
        conn = self.pg_store._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM nodes 
                    WHERE id = ANY(%s::uuid[])
                """, (node_ids,))
                records = cur.fetchall()
                
                nodes = []
                for rec in records:
                    try:
                        clean_rec = {k: v for k, v in rec.items() if v is not None}
                        # Inject missing fields to fulfill the validation schema contract
                        if "processor_name" not in clean_rec:
                            clean_rec["processor_name"] = "Ingestion_Parser_Fallback"
                        if "processor_version" not in clean_rec:
                            clean_rec["processor_version"] = "1.0.0"
                            
                        nodes.append(Node(**clean_rec))
                    except Exception as e:
                        logger.error(f"Failed to parse node {rec.get('id')}: {e}")
                return nodes
        finally:
            self.pg_store._put_conn(conn)
            
    ContextHydrator._fetch_nodes = patched_fetch_nodes
    print("🩹 ContextHydrator pydantic validation patch applied successfully.")

# ==========================================
# Execution Wrappers
# ==========================================

async def execute_routing(router: QueryRouter, user_query: str, tenant_id: str) -> Optional[RetrievalPlan]:
    if RICH_AVAILABLE:
        print("\n[bold magenta]🧠 Phase 4.1: Routing (SLM)[/bold magenta]")
    else:
        print("\n🧠 Phase 4.1: Routing (SLM)")
        
    plan = await router.route(user_query=user_query, tenant_id=tenant_id)
    
    if RICH_AVAILABLE:
        plan_json = json.dumps(plan.model_dump(), indent=2)
        console.print(Panel(plan_json, title="📝 Retrieval Plan Generated", border_style="green"))
    else:
        print("\n📝 Retrieval Plan Generated:\n", json.dumps(plan.model_dump(), indent=2))
        
    return plan

async def execute_retrieval(retriever: GraphRetriever, plan: RetrievalPlan, tenant_id: str) -> Dict[str, Any]:
    if RICH_AVAILABLE:
        print("\n[bold cyan]🕸️  Phase 4.2: Graph & Vector Retrieval[/bold cyan]")
    else:
        print("\n🕸️  Phase 4.2: Graph & Vector Retrieval")
        
    result = await retriever.retrieve(plan=plan, tenant_id=tenant_id)
    
    if RICH_AVAILABLE:
        table = Table(title="🔍 Retrieval Structural Match Count")
        table.add_column("Sub-Graph Component", style="magenta")
        table.add_column("Count Extracted", style="green")
        table.add_row("Unique Nodes", str(len(result["node_ids"])))
        table.add_row("Tracked Graph Edges", str(len(result["edge_ids"])))
        console.print(table)
    else:
        print(f"   Nodes Extracted: {len(result['node_ids'])}, Edges Extracted: {len(result['edge_ids'])}")
        
    return result

async def execute_hydration(pg_store: PostgresStore, retrieval_result: Dict[str, Any]) -> str:
    if RICH_AVAILABLE:
        print("\n[bold green]🧬 Phase 4.3: Context Hydration & Prompt Engineering (Postgres Store)[/bold green]")
    else:
        print("\n🧬 Phase 4.3: Context Hydration & Prompt Engineering (Postgres Store)")
        
    hydrator = ContextHydrator(pg_store=pg_store)
    context_string = await hydrator.hydrate(retrieval_result)
    
    if RICH_AVAILABLE:
        console.print(Panel(context_string, title="📄 Final Formatted LLM Context Block", border_style="blue", padding=(1, 2)))
    else:
        print("\n--- Final Formatted LLM Context Block ---")
        print(context_string)
        print("------------------------------------------")
        
    return context_string

# ==========================================
# Interactive Pipeline Engine
# ==========================================

def get_menu_selection() -> str:
    print("\n🛠️  SpatialRAG Query Strategy Test Menu")
    print("1. Phase 4.1 Only: Query Routing Plan Generator")
    print("2. Phase 4.2 & 4.3: Retrieve & Hydrate Context (Uses Mock Route Plan)")
    print("3. Full Run: Execute 4.1 Routing -> 4.2 Retrieval -> 4.3 Hydration -> Synthesis Response")
    
    return input("Select operation mode (1, 2, or 3): ").strip()

async def interactive_test_loop(router: QueryRouter, retriever: GraphRetriever, pg_store: PostgresStore, tenant_id: str):
    mode = get_menu_selection()
    if mode not in ['1', '2', '3']:
        print("❌ Invalid selection. Defaulting to Full Run (Mode 3).")
        mode = '3'
        
    print(f"\n🚀 Interactive Engine Active (Mode {mode}). Type 'exit' or 'q' to end session.")
    
    while True:
        try:
            if RICH_AVAILABLE:
                user_query = console.input("\n[bold white]👤 Enter Test Query > [/bold white]").strip()
            else:
                user_query = input("\n👤 Enter Test Query > ").strip()
                
            if user_query.lower() in ['exit', 'quit', 'q']:
                break
            if not user_query:
                continue
            
            if mode == '1':
                await execute_routing(router, user_query, tenant_id)
                
            elif mode == '2':
                fallback_plan = RetrievalPlan(
                    intent_summary=user_query,
                    query_type="specific_lookup",
                    target_modalities=["textual_content", "table_container"],
                    target_granularities=["element", "block", "section"],
                    filter_edges=["hierarchy", "reference", "spatial_relation"],
                    high_confidence_node_ids=[],
                    use_community_nodes=False
                )
                retrieved_context = await execute_retrieval(retriever, fallback_plan, tenant_id)
                await execute_hydration(pg_store, retrieved_context)
                
            elif mode == '3':
                plan = await execute_routing(router, user_query, tenant_id)
                if plan:
                    retrieved_context = await execute_retrieval(retriever, plan, tenant_id)
                    final_context_str = await execute_hydration(pg_store, retrieved_context)
                    # Pass the router client to use the LLM connection info
                    await run_synthesis_phase(router, final_context_str, user_query)
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Pipeline Crash Intercepted: {e}")
            import traceback
            traceback.print_exc()

# ==========================================
# Main Orchestrator Entrypoint
# ==========================================

async def main():
    print("🛠️  SpatialRAG Multi-Phase Inference Evaluator")
    print("==============================================")
    
    # Run patch before initializing data structures
    patch_context_hydrator()
    
    config = EngineConfig()
    
    router = await init_router(config)
    retriever = await init_retriever(config)
    pg_store = PostgresStore(config=config.db)
    
    target_tenant = "test_tenant"
    
    try:
        await interactive_test_loop(router, retriever, pg_store, target_tenant)
    finally:
        print("\n🔒 Tearing down operational database connections safely...")
        await retriever.close()
        print("✅ Shutdown routine completed cleanly.")

if __name__ == "__main__":
    import logging
    # Initialize basic logging to avoid missing logger reference crashes inside the patch
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("core.query.hydrator")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)