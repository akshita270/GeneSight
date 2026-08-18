from __future__ import annotations
from neo4j import GraphDatabase
from config import settings
import logging

logger = logging.getLogger("genesight")

# ── Singleton driver — one connection pool for the process lifetime ───────────
_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        uri = settings.neo4j_uri
        if not uri:
            raise RuntimeError("NEO4J_URI is not configured")
        _driver = GraphDatabase.driver(
            uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        logger.info("Neo4j driver initialised (uri=%s)", uri)
    return _driver


class Neo4jClient:
    """
    Thin wrapper around the process-level Neo4j driver.
    All write operations on the graph should go through `batch_upsert_graph()`
    which does everything in a single round-trip with UNWIND instead of N+1
    individual merge_node / merge_edge calls.
    """

    @property
    def driver(self):
        return _get_driver()

    # ── Batch write — replaces N+1 individual merge calls ────────────────────

    def batch_upsert_graph(
        self,
        disease: str,
        genes: list[str],
        gene_props: dict[str, dict],   # gene_name → {ncbi_id, uniprot_id, function}
    ) -> None:
        """
        Upsert the disease node, all gene nodes, and gene→disease edges in
        a single transaction using UNWIND — O(1) round trips regardless of
        how many genes there are.
        """
        gene_rows = [
            {
                "name": g,
                **{k: v for k, v in gene_props.get(g, {}).items() if v is not None},
            }
            for g in genes
        ]
        with _get_driver().session() as s:
            s.run(
                """
                MERGE (d:Disease {name: $disease})
                WITH d
                UNWIND $genes AS row
                  MERGE (g:Gene {name: row.name})
                  SET g += row
                  MERGE (g)-[:ASSOCIATED_WITH]->(d)
                """,
                disease=disease,
                genes=gene_rows,
            )

    # ── Legacy single-entity helpers (used by health check + tests) ───────────

    def merge_node(self, label: str, name: str, props: dict | None = None) -> None:
        props = props or {}
        with _get_driver().session() as s:
            s.run(
                f"MERGE (n:{label} {{name: $name}}) SET n += $props",
                name=name, props=props,
            )

    def merge_edge(self, src: str, tgt: str, relation: str, props: dict | None = None) -> None:
        props = props or {}
        with _get_driver().session() as s:
            s.run(
                f"""
                MATCH (a {{name: $src}}), (b {{name: $tgt}})
                MERGE (a)-[r:{relation}]->(b)
                SET r += $props
                """,
                src=src, tgt=tgt, props=props,
            )

    def get_neighbors(self, name: str) -> list[dict]:
        with _get_driver().session() as s:
            result = s.run(
                "MATCH (n {name: $name})-[r]-(m) RETURN m.name AS neighbor, type(r) AS relation",
                name=name,
            )
            return [dict(rec) for rec in result]

    def get_full_graph(self) -> dict:
        with _get_driver().session() as s:
            nodes = s.run("MATCH (n) RETURN labels(n)[0] AS type, n.name AS name")
            edges = s.run("MATCH (a)-[r]->(b) RETURN a.name AS src, type(r) AS rel, b.name AS tgt")
            return {
                "nodes": [dict(r) for r in nodes],
                "edges": [dict(r) for r in edges],
            }

    def clear(self) -> None:
        with _get_driver().session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        """No-op — driver lifetime is managed at process level."""
