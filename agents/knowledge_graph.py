from __future__ import annotations
from db.neo4j_client import Neo4jClient
from models.schemas import GraphNode, GraphEdge, KnowledgeGraph


class KnowledgeGraphAgent:
    async def run(self, entities: dict, db_data: list[dict]) -> KnowledgeGraph:
        disease = entities["diseases"][0] if entities["diseases"] else "Unknown Disease"
        genes   = entities.get("genes", [])

        # Build per-gene property map from genomics DB enrichment data
        gene_props: dict[str, dict] = {}
        for entry in db_data:
            gene = entry.get("gene")
            if gene:
                gene_props[gene] = {
                    "ncbi_id":    entry.get("ncbi_id"),
                    "uniprot_id": entry.get("uniprot_id"),
                    "function":   entry.get("function", ""),
                }

        # One batched UNWIND transaction instead of N+1 individual round trips
        db = Neo4jClient()
        db.batch_upsert_graph(disease, genes, gene_props)

        # Build in-memory graph for downstream agents
        nodes: list[GraphNode] = [GraphNode(id=disease, label=disease, type="disease")]
        edges: list[GraphEdge] = []

        for g in genes:
            nodes.append(GraphNode(id=g, label=g, type="gene"))
            edges.append(GraphEdge(source=g, target=disease, relation="ASSOCIATED_WITH", confidence=0.7))

        for p in entities.get("proteins", []):
            nodes.append(GraphNode(id=p, label=p, type="protein"))

        return KnowledgeGraph(nodes=nodes, edges=edges)
