from __future__ import annotations
from db.neo4j_client import Neo4jClient
from models.schemas import GraphNode, GraphEdge, KnowledgeGraph

class KnowledgeGraphAgent:
    async def run(self, entities: dict, db_data: list[dict]) -> KnowledgeGraph:
        db = Neo4jClient()
        try:
            nodes, edges = [], []

            # Primary disease node
            disease = entities["diseases"][0] if entities["diseases"] else "Unknown Disease"
            nodes.append(GraphNode(id=disease, label=disease, type="disease"))
            db.merge_node("Disease", disease)

            # Gene nodes + edges to disease
            for g in entities["genes"]:
                nodes.append(GraphNode(id=g, label=g, type="gene"))
                db.merge_node("Gene", g)
                edges.append(GraphEdge(
                    source=g, target=disease,
                    relation="ASSOCIATED_WITH", confidence=0.7
                ))
                db.merge_edge(g, disease, "ASSOCIATED_WITH")

            # Protein nodes
            for p in entities["proteins"]:
                nodes.append(GraphNode(id=p, label=p, type="protein"))
                db.merge_node("Protein", p)

            # Enrich with DB data — add UniProt function as property
            for entry in db_data:
                gene = entry.get("gene")
                if gene:
                    db.merge_node("Gene", gene, {
                        "ncbi_id": entry.get("ncbi_id"),
                        "uniprot_id": entry.get("uniprot_id"),
                        "function": entry.get("function", ""),
                    })
        finally:
            db.close()

        return KnowledgeGraph(nodes=nodes, edges=edges)
