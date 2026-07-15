# SpatialRAG: A Local-First, Multimodal Graph RAG Engine (WIP demo included)

## Project Overview

**SpatialRAG** is an innovative **local-first, multimodal Graph RAG (Retrieval Augmented Generation) engine** designed to ingest and process complex documents, including PDFs, engineering drawings, videos, and audio. It transforms these diverse data types into a unified knowledge graph, enabling advanced cross-modal retrieval and question-answering capabilities. The system achieves this by representing various entities as nodes and their relationships as edges, further organizing complex internal entities into subgraphs.

## 📺 Demo & System Walkthrough

Watch SpatialRAG process a complex technical query, routing through the graph and synthesizing an accurate response.

<video src="https://github.com/user-attachments/assets/3a2e97a5-3d7c-45be-837c-2fa7f239d9d8" controls="controls" muted="muted" style="max-width: 100%; height: auto;"></video>

 Here is a technical breakdown of exactly how the system processes that query:

### Interactive Query Session
The following walkthrough demonstrates the multi-phase inference engine in action, processing a complex technical query about the CRNN model's performance.

#### **User Query:** 
> *"What is detailed performance metric of crnn on iit5k dataset?"*

<details>
<summary><b>Phase 4.1: Routing (SLM) - <i>Click to expand</i></b></summary>

The system first analyzes the query to generate a structured retrieval plan. It identifies the target modalities (tables and text) and granularities (elements and blocks) needed to answer the question.

```json
{
  "intent_summary": "CRNN IIT5K dataset performance metric",
  "query_type": "specific_lookup",
  "target_modalities": ["table_container", "textual_content"],
  "target_granularities": ["element", "block"],
  "filter_edges": ["hierarchy"],
  "high_confidence_node_ids": [
    "2d184d0c-f753-4a45-ab2e-dca52d21201c",
    "863a4480-7aee-49f7-87d0-494d70c8c54c",
    "d7e19e3a-37ca-48ca-9b6a-089b165c05a3"
  ],
  "high_confidence_weight": 1.5,
  "use_community_nodes": false
}
```
</details>

<details>
<summary><b>Phase 4.2: Graph & Vector Retrieval - <i>Click to expand</i></b></summary>

The retriever executes the plan by performing a hybrid search across Qdrant (vector) and Neo4j (graph). It expands the search to capture related context and then prunes it to the most relevant nodes.

| Sub-Graph Component | Count Extracted |
|---------------------|-----------------|
| Unique Nodes        | 10              |
| Tracked Graph Edges | 7               |

*Retrieval complete. Returning 10 nodes and 7 edges.*
</details>

<details>
<summary><b>Phase 4.3: Context Hydration - <i>Click to expand</i></b></summary>

The system fetches the full content from PostgreSQL and formats it into a structured context block for the LLM, including relevant table data and textual descriptions.

**Extracted Context Snippet:**
> `[Table Container]: Table 2. Recognition accuracies (%) on four datasets.`
> `| Method | IIIT5k (50) | IIIT5k (1k) | IIIT5k (None) |`
> `|--------|-------------|-------------|---------------|`
> `| CRNN   | 97.6        | 94.4        | 78.2          |`
</details>

#### **🎯 Synthesized Answer:**
Based on the provided context, the detailed performance metrics of the **CRNN (Convolutional Recurrent Neural Network)** model on the **IIIT 5k-word (IIIT5k)** dataset are as follows:

*   **With 50-word lexicon:** **97.6%** accuracy.
*   **With 1k-word lexicon:** **94.4%** accuracy.
*   **Without lexicon (None):** **78.2%** accuracy.

The CRNN model consistently outperforms most state-of-the-art approaches on the IIIT5k dataset, particularly in constrained lexicon cases.

---

## Architecture and Key Features

SpatialRAG's architecture is built around a robust pipeline that handles data ingestion, graph construction, enrichment, embedding, and retrieval. Key components and features include:

### 1. Custom Graph Schema (core/models)

The project utilizes a custom schema for document nodes and edges, designed for flexibility and comprehensive information capture. This schema includes attributes such as `modality`, `modality_category`, `edge_type`, and `edge_type_category`, allowing for diverse data types and relationships. It also incorporates `subgraph_tenant_ids` and `processor_versions` for enhanced organization and future updates. A crucial aspect is the full schema version control, facilitating easy dataset updates.

### 2. Document Parsing with Docling (core/parsers)

Docling is employed as the primary document parsing engine due to its advanced capabilities and upgradability. It extracts bounding box (bbox) information for all entities within documents and leverages structured headings to form a general hierarchy. This process generates various nodes and edges, mapping them to the custom schema fields and forming initial subgraphs.

### 3. Post-Processing Strategies (core/processors)

After initial parsing, several post-processing strategies are applied to refine the nodes and edges:

- **Spatial Edges**: Formed using bounding box information to capture spatial relationships between entities.

- **Contextual Prepending**: Headings are prepended to paragraphs to provide richer context.

- **Relation Edges**: Basic regex is used to form initial relation edges.

- **Evidence Part of Edges**: Mechanisms for forming evidence associated with edges are implemented.

### 4. AI Enrichment

Nodes and edges undergo AI enrichment, including:

- **VLM/LLM Explanations**: Visual Language Models (VLMs) and Large Language Models (LLMs) are used to explain images or diagrams, populating the `subgraph_head` with summarized data.

### 5. Embedding & Retrieval Strategy (core/embeddings): The 3-Pillar Architecture

SpatialRAG employs a sophisticated three-pillar embedding and retrieval strategy to manage the complexity of graph data efficiently:

| Pillar | Description | Storage | Embedded? |
| --- | --- | --- | --- |
| **Pillar A (Entity)** | Raw content (text block, image path, table string) | PostgreSQL, Neo4j (as node), Qdrant | **YES** – Node content is embedded |
| **Pillar B (Topology)** | Structural skeleton (source → target pointers) | PostgreSQL, Neo4j (as relationship) | **NO** – Traversed via Neo4j |
| **Pillar C (Evidence)** | Localized rationale explaining *why* an edge exists | PostgreSQL, Qdrant | **YES** – Edge evidence text is embedded |

This approach avoids embedding every edge directly, which would lead to an explosion in vector space. Instead, only the evidence text (Pillar C) for each edge is embedded. During retrieval, the process involves:

1. **Land**: User query performs a vector search on Node content to get seed nodes.

1. **Walk**: Neo4j (Pillar B) is traversed to identify candidate edges.

1. **Rerank/Filter**: Qdrant's `evidence_vectors` are searched with the query, filtered by candidate edge IDs, to semantically rerank paths based on the meaning of the connection.

1. **Hydrate**: Full content is fetched from PostgreSQL.

### 6. SLM-Driven Query Routing (core/query)

A **Semantic Relation Processor (SLM)** runs after layout parsing to generate semantic edges (e.g., `valve_v101 → reactor_r1` with `type = "controls_flow_to"` and `evidence = "Valve V-101 controls water flow to Reactor R-1"`). These edges are stored in PostgreSQL and Neo4j, with their evidence embedded in Qdrant.

The **Query Planning (SLM-Driven Router)** intercepts user queries, feeds them to a lightweight SLM with available `type_category` values, and generates a routing plan (e.g., "only traverse PHYSICAL_CONNECTION edges"). The SLM then executes the plan: vector search → graph traversal filtered by the plan → reranking → hydration.

### 7. Storage (core/stores)

All generated nodes and edges are saved in a multi-modal storage system consisting of:

- **PostgreSQL**: For structured data and full content storage.

- **Neo4j**: For graph topology and relationships.

- **Qdrant**: For vector embeddings of nodes and edge evidence.

## Current Status and Future Work

### Implemented:

- Document parsing and custom schema mapping.

- Node and edge creation with initial hierarchy and spatial information.

- Basic AI enrichment for nodes and edges.

- Three-pillar embedding and retrieval strategy.

- SLM-driven query routing.

- Integration with PostgreSQL, Neo4j, and Qdrant for storage.

- A `test.py` script demonstrates the complete pipeline from PDF parsing to embedding and storage.

- A `test_query.py` script showcases the basic RAG pipeline for querying the document.

### Future Work:

- Implementation of a spatial relationship extractor.

- Enhanced understanding and containerization of complex subgraphs (e.g., engineering diagrams).

- Advanced community detection similar to Microsoft Graph RAG.

- Further research and testing of alternative embedding strategies for evidence text.

## Getting Started

### Prerequisites

- Docker and Docker Compose

- Python 3.9+

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/idkjashan/Spatial-rag.git
   cd Spatial-rag
   ```

1. **Set up environment variables:**

   Create a `.env` file based on `compose.yml` for database credentials and API keys.

1. **Build and Start Docker services:**

   ```bash
   docker build .
   docker compose up -d
   ```

### Running the Pipeline Demo

To run the full ingestion pipeline on a sample PDF (`CRNN.pdf` ):

```bash
 docker-compose exec app python test.py
```

This script will parse the PDF, extract nodes and edges, post-process, enrich, embed, and store them. It also generates `enriched_output_full.json` and use visualize_graph.py to generate `all_pages.png` for visualization.

### Running the Query Demo

To test the retrieval and query capabilities:

```bash
 docker-compose exec app python test_query.py
```

This script demonstrates how to query the indexed document and retrieve relevant information.

## Project Structure

```
. Spatial-rag/
├── CRNN.pdf
├── Dockerfile
├── all_pages.png
├── compose.yml
├── core/
│   ├── config.py
│   ├── embeddings/
│   │   └── graph_embedder.py
│   ├── models/
│   │   ├── base.py
│   │   ├── document.py
│   │   ├── edge.py
│   │   └── node.py
│   │   └── processor.py
│   ├── parsers/
│   │   ├── base.py
│   │   ├── docling_mapper.py
│   │   └── docling_parser.py
│   ├── pipeline.py
│   ├── processors/
│   │   ├── contextual_enricher.py
│   │   └── post_processor.py
│   ├── query/
│   │   ├── hydrator.py
│   │   ├── retriever.py
│   │   └── router.py
│   └── stores/
│       ├── neo4j_store.py
│       ├── postgres_store.py
│       ├── qdrant_store.py
│       └── storage_manager.py
├── enriched_output_full.json
├── requirements.txt
├── test.py
├── test_query.py
└── visualize_graph.py
```

## Technologies Used

- **Python**: Core programming language.

- **Pydantic**: Data validation and settings management.

- **Docling**: Document parsing.

- **PostgreSQL**: Relational database for structured data.

- **Neo4j**: Graph database for relationships.

- **Qdrant**: Vector database for embeddings.

- **OpenAI API**: For LLM/VLM enrichment and embeddings.

- **Docker & Docker Compose**: For local development environment setup.

## Author

Jashan :)
