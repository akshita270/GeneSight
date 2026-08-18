from __future__ import annotations
from neo4j import GraphDatabase
from config import settings


class Neo4jClient:
    def __init__(self):
        uri = settings.neo4j_uri
        if not uri:
            raise RuntimeError("NEO4J_URI is not configured")
        self.driver = GraphDatabase.driver(uri, auth=(settings.neo4j_user, settings.neo4j_password))

    def merge_node(self, label: str, name: str, props: dict = None):
        props = props or {}
        with self.driver.session() as s:
            s.run(
                f"MERGE (n:{label} {{name: $name}}) SET n += $props",
                name=name, props=props
            )

    def merge_edge(self, src: str, tgt: str, relation: str, props: dict = None):
        props = props or {}
        with self.driver.session() as s:
            s.run(f"""
                MATCH (a {{name: $src}}), (b {{name: $tgt}})
                MERGE (a)-[r:{relation}]->(b)
                SET r += $props
            """, src=src, tgt=tgt, props=props)

    def get_neighbors(self, name: str) -> list[dict]:
        with self.driver.session() as s:
            result = s.run("""
                MATCH (n {name: $name})-[r]-(m)
                RETURN m.name AS neighbor, type(r) AS relation
            """, name=name)
            return [dict(rec) for rec in result]

    def get_full_graph(self) -> dict:
        with self.driver.session() as s:
            nodes = s.run("MATCH (n) RETURN labels(n)[0] AS type, n.name AS name")
            edges = s.run("MATCH (a)-[r]->(b) RETURN a.name AS src, type(r) AS rel, b.name AS tgt")
            return {
                "nodes": [dict(r) for r in nodes],
                "edges": [dict(r) for r in edges],
            }

    def clear(self):
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def close(self):
        self.driver.close()