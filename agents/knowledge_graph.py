from db.neo4j_client import Neo4jClient
from models.schemas import GraphNode, GraphEdge, KnowledgeGraph

class KnowledgeGraphAgent:
    def __init__(self):
        self.db = Neo4jClient()

    async def run(self, entities: dict, db_data: list[dict]) -> KnowledgeGraph:
        nodes, edges = [], []

        # Primary disease node
        disease = entities["diseases"][0] if entities["diseases"] else "Unknown Disease"
        nodes.append(GraphNode(id=disease, label=disease, type="disease"))
        self.db.merge_node("Disease", disease)

        # Gene nodes + edges to disease
        for g in entities["genes"]:
            nodes.append(GraphNode(id=g, label=g, type="gene"))
            self.db.merge_node("Gene", g)
            edges.append(GraphEdge(
                source=g, target=disease,
                relation="ASSOCIATED_WITH", confidence=0.7
            ))
            self.db.merge_edge(g, disease, "ASSOCIATED_WITH")

        # Protein nodes
        for p in entities["proteins"]:
            nodes.append(GraphNode(id=p, label=p, type="protein"))
            self.db.merge_node("Protein", p)

        # Enrich with DB data — add UniProt function as property
        for entry in db_data:
            gene = entry.get("gene")
            if gene:
                self.db.merge_node("Gene", gene, {
                    "ncbi_id": entry.get("ncbi_id"),
                    "uniprot_id": entry.get("uniprot_id"),
                    "function": entry.get("function", ""),
                })

        return KnowledgeGraph(nodes=nodes, edges=edges)

    def close(self):
        self.db.close()