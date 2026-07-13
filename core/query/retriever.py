import math
import logging
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
from openai import AsyncOpenAI

from core.query.router import RetrievalPlan
from core.config import DatabaseConfig

logger = logging.getLogger(__name__)

class GraphRetriever:
    """
    Phase 4.2: Executes the RetrievalPlan against Neo4j and Qdrant.
    Implements the "Land, Expand, Multi-Signal Score" strategy.
    """
    
    # Static heuristic scores for edges without vectors (Signal B Fallback)
    EDGE_HEURISTICS = {
        "reference": 0.8,
        "semantic_relation": 0.8,
        "physical_connection": 0.7,
        "flow_direction": 0.7,
        "caption": 0.6,
        "spatial_relation": 0.5,
        "hierarchy": 0.4,
        "table_hierarchy": 0.4,
        "list_hierarchy": 0.4,
        "read_order": 0.1 # Penalize read_order heavily
    }

    def __init__(
        self, 
        db_config: DatabaseConfig,
        qdrant_node_collection: str = "nodes",
        qdrant_edge_collection: str = "edges",
        embed_base_url: str = "http://localhost:11434/v1",
        embed_api_key: str = "ollama",
        embedding_model: str = "nomic-embed-text",
        max_final_nodes: int = 10,
        min_expand_threshold: int = 10
    ):
        self.db_config = db_config
        self.qdrant_node_collection = qdrant_node_collection
        self.qdrant_edge_collection = qdrant_edge_collection
        
        self.embedding_model = embedding_model
        self.embed_client = AsyncOpenAI(base_url=embed_base_url, api_key=embed_api_key, timeout=30.0, max_retries=0)
        
        self.max_final_nodes = max_final_nodes
        self.min_expand_threshold = min_expand_threshold
        
        # Initialize DB drivers
        self.neo4j_driver = AsyncGraphDatabase.driver(
            db_config.neo4j_url, 
            auth=(db_config.neo4j_user, db_config.neo4j_password)
        )
        self.qdrant_client = AsyncQdrantClient(url=db_config.qdrant_url, api_key=db_config.qdrant_api_key)

    async def close(self):
        await self.neo4j_driver.close()
        await self.qdrant_client.close()

    async def retrieve(self, plan: RetrievalPlan, tenant_id: str = "default", doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Main execution pipeline."""
        logger.info(f"Starting retrieval for intent: '{plan.intent_summary}'")
        
        # 1. Embed the intent summary
        query_vector = await self._embed_text(plan.intent_summary)
        
        # 2. LAND: Get Seed Nodes from Qdrant
        seed_nodes = await self._land_seed_nodes(query_vector, plan, tenant_id, doc_id)
        if not seed_nodes:
            logger.warning("No seed nodes found in Qdrant.")
            return {"nodes": [], "edges": []}
            
        # 3. EXPAND: Traverse Neo4j
        candidate_nodes, candidate_edges = await self._expand_subgraph(seed_nodes, plan, tenant_id, doc_id)
        logger.info(f"Expanded to {len(candidate_nodes)} nodes and {len(candidate_edges)} edges.")
        
        # 4. SCORE: Multi-Signal Reranking
        scored_nodes = await self._score_candidates(query_vector, candidate_nodes, candidate_edges, seed_nodes, plan)
        
        # 5. PRUNE: Select Top M and inject parents
        final_node_ids, final_edge_ids = self._prune_and_inject(scored_nodes, candidate_edges, plan)
        
        logger.info(f"Retrieval complete. Returning {len(final_node_ids)} nodes and {len(final_edge_ids)} edges.")
        return {
            "node_ids": list(final_node_ids),
            "edge_ids": list(final_edge_ids)
        }

    # ==========================================
    # Step 1: Land
    # ==========================================
    async def _land_seed_nodes(self, query_vector: List[float], plan: RetrievalPlan, tenant_id: str, doc_id: Optional[str]) -> List[Dict]:
        """Queries Qdrant for seed nodes based on the retrieval plan filters."""
        must_filters = [
            {"key": "tenant_id", "match": {"value": tenant_id}},
            {"key": "modality_category", "match": {"any": plan.target_modalities}},
            {"key": "granularity", "match": {"any": plan.target_granularities}}
        ]
        if doc_id:
            must_filters.append({"key": "doc_id", "match": {"value": doc_id}})
            
        results = await self.qdrant_client.search(
            collection_name=self.qdrant_node_collection,
            query_vector=query_vector,
            query_filter={"must": must_filters},
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        
        seed_nodes = []
        for point in results:
            seed_nodes.append({
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload or {}
            })
            
        # Inject high confidence nodes if not already present
        existing_ids = {n["id"] for n in seed_nodes}
        for hc_id in plan.high_confidence_node_ids:
            if hc_id not in existing_ids:
                # Fetch payload from Qdrant directly
                try:
                    points = await self.qdrant_client.retrieve(
                        collection_name=self.qdrant_node_collection,
                        ids=[hc_id],
                        with_payload=True
                    )
                    if points:
                        seed_nodes.append({
                            "id": str(points[0].id),
                            "score": 0.5, # Base score for injected TOC nodes
                            "payload": points[0].payload or {}
                        })
                except Exception:
                    pass
                    
        return seed_nodes

    # ==========================================
    # Step 2: Expand
    # ==========================================
    async def _expand_subgraph(self, seed_nodes: List[Dict], plan: RetrievalPlan, tenant_id: str, doc_id: Optional[str]) -> Tuple[Dict, List[Dict]]:
        """Executes priority and fallback traversals in Neo4j."""
        seed_ids = [n["id"] for n in seed_nodes]
        candidate_nodes = {n["id"]: n for n in seed_nodes}
        candidate_edges = []
        
        async with self.neo4j_driver.session() as session:
            # Layer 1: Priority Edges
            query1 = """
            UNWIND $seed_ids AS seed_id
            MATCH (s:Node {id: seed_id})-[r:CONNECTS]-(t:Node)
            WHERE r.type_category IN $filter_edges 
              AND t.tenant_id = $tenant_id
              AND ($doc_id IS NULL OR t.doc_id = $doc_id)
            RETURN s.id AS source_id, t.id AS target_id, r.id AS edge_id, r.type_category AS edge_type
            """
            result = await session.run(query1, {
                "seed_ids": seed_ids, 
                "filter_edges": plan.filter_edges, 
                "tenant_id": tenant_id,
                "doc_id": doc_id
            })
            
            records = await result.data()
            for rec in records:
                candidate_edges.append(rec)
                if rec["target_id"] not in candidate_nodes:
                    candidate_nodes[rec["target_id"]] = {"id": rec["target_id"], "score": 0.0, "payload": {}}
                    
            # Layer 2: Fallback Expansion (if priority fan-out was too small)
            if len(candidate_nodes) < self.min_expand_threshold:
                logger.info("Priority fan-out too small. Executing fallback expansion...")
                query2 = """
                UNWIND $seed_ids AS seed_id
                MATCH (s:Node {id: seed_id})-[r:CONNECTS]-(t:Node)
                WHERE r.type_category <> 'read_order' 
                  AND t.tenant_id = $tenant_id
                  AND ($doc_id IS NULL OR t.doc_id = $doc_id)
                RETURN s.id AS source_id, t.id AS target_id, r.id AS edge_id, r.type_category AS edge_type
                """
                result2 = await session.run(query2, {
                    "seed_ids": seed_ids, 
                    "tenant_id": tenant_id,
                    "doc_id": doc_id
                })
                
                records2 = await result2.data()
                for rec in records2:
                    # Avoid duplicate edges
                    if not any(e["edge_id"] == rec["edge_id"] for e in candidate_edges):
                        candidate_edges.append(rec)
                    if rec["target_id"] not in candidate_nodes:
                        candidate_nodes[rec["target_id"]] = {"id": rec["target_id"], "score": 0.0, "payload": {}}
                        
        return candidate_nodes, candidate_edges

    # ==========================================
    # Step 3: Multi-Signal Score
    # ==========================================
    async def _score_candidates(self, query_vector: List[float], candidate_nodes: Dict, candidate_edges: List[Dict], seed_nodes: List[Dict], plan: RetrievalPlan) -> List[Dict]:
        """Calculates the composite relevance score for all candidate nodes."""
        
        # 1. Fetch expanded node vectors from Qdrant to compute Signal A
        expanded_ids = [nid for nid in candidate_nodes.keys() if nid not in {s["id"] for s in seed_nodes}]
        expanded_vectors = {}
        if expanded_ids:
            points = await self.qdrant_client.retrieve(
                collection_name=self.qdrant_node_collection,
                ids=expanded_ids,
                with_payload=True,
                with_vectors=True
            )
            for p in points:
                expanded_vectors[str(p.id)] = {
                    "vector": p.vector,
                    "payload": p.payload or {}
                }
                
        # 2. Fetch edge vectors from Qdrant for Signal B (Batch retrieve)
        edge_ids = [e["edge_id"] for e in candidate_edges if e.get("edge_id")]
        edge_vectors = {}
        if edge_ids:
            # Qdrant retrieve is limited to 1000 IDs, safe for our subgraph size
            e_points = await self.qdrant_client.retrieve(
                collection_name=self.qdrant_edge_collection,
                ids=edge_ids,
                with_payload=True,
                with_vectors=True
            )
            for p in e_points:
                edge_vectors[str(p.id)] = p.vector

        # 3. Calculate Edge Scores
        edge_scores = {}
        incoming_edges = defaultdict(list)
        
        for edge in candidate_edges:
            e_id = edge.get("edge_id")
            if not e_id: continue
            
            # Signal B: Vector score or Heuristic
            if e_id in edge_vectors:
                e_score = self._cosine_sim(query_vector, edge_vectors[e_id])
            else:
                e_score = self.EDGE_HEURISTICS.get(edge.get("edge_type", ""), 0.3)
                
            edge_scores[e_id] = e_score
            incoming_edges[edge["target_id"]].append(e_score)
            
        # 4. Calculate Node Scores
        scored_nodes = []
        high_conf_set = set(plan.high_confidence_node_ids)
        
        for n_id, node in candidate_nodes.items():
            # Signal A: Node Vector Similarity
            if n_id in {s["id"] for s in seed_nodes}:
                # Use score from initial Qdrant search
                node_score = next((s["score"] for s in seed_nodes if s["id"] == n_id), 0.0)
            elif n_id in expanded_vectors:
                # Compute cosine sim for expanded nodes
                node_score = self._cosine_sim(query_vector, expanded_vectors[n_id]["vector"])
                # Update payload if it was missing
                node["payload"] = expanded_vectors[n_id].get("payload", {})
            else:
                node_score = 0.0 # Unknown node
                
            # Signal C: Graph Centrality (Hub Boost)
            centrality_boost = min(len(incoming_edges[n_id]) * 0.1, 0.3)
            
            # Max incoming edge score
            max_edge_score = max(incoming_edges[n_id]) if incoming_edges[n_id] else 0.0
            
            # Final Score Formula
            final_score = node_score + (max_edge_score * 0.5) + centrality_boost
            
            # Signal D: High Confidence Boost
            if n_id in high_conf_set:
                final_score *= plan.high_confidence_weight
                
            scored_nodes.append({
                "id": n_id,
                "score": final_score,
                "parent_id": node.get("payload", {}).get("parent_id") or node.get("parent_id")
            })
            
        # Sort descending
        scored_nodes.sort(key=lambda x: x["score"], reverse=True)
        return scored_nodes

    # ==========================================
    # Step 4: Prune & Inject
    # ==========================================
    def _prune_and_inject(self, scored_nodes: List[Dict], candidate_edges: List[Dict], plan: RetrievalPlan) -> Tuple[set, set]:
        """Selects top M nodes, injects contextual parents, and filters edges."""
        
        # 1. Select Top M nodes
        top_nodes = scored_nodes[:self.max_final_nodes]
        final_node_ids = {n["id"] for n in top_nodes}
        
        # 2. Contextual Parent Injection
        # If a selected node is an element/block, ensure its parent section/table is included
        for node in top_nodes:
            parent_id = node.get("parent_id")
            if parent_id and parent_id not in final_node_ids:
                # Check if parent is in candidate pool to avoid blind DB calls here
                # For simplicity, we add it. Hydrator will fetch it from Postgres.
                final_node_ids.add(parent_id)
                logger.debug(f"Injected parent {parent_id} for node {node['id']}")
                
        # 3. Filter Edges
        final_edge_ids = set()
        for edge in candidate_edges:
            # Keep edge if both source and target are in the final node set
            if edge["source_id"] in final_node_ids and edge["target_id"] in final_node_ids:
                if edge.get("edge_id"):
                    final_edge_ids.add(edge["edge_id"])
                    
        return final_node_ids, final_edge_ids

    # ==========================================
    # Utility Methods
    # ==========================================
    async def _embed_text(self, text: str) -> List[float]:
        """Embeds the text using the configured embedder."""
        response = await self.embed_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def _cosine_sim(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates cosine similarity between two vectors."""
        if not vec1 or not vec2: return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        if mag1 == 0 or mag2 == 0: return 0.0
        return dot_product / (mag1 * mag2)