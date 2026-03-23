from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

URI      = os.getenv("NEO4J_URI")
USER     = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to Neo4j at {URI}...")
print(f"Using username: {USER}")

try:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("Connected successfully!\n")

    with driver.session() as s:
        # Create sample nodes
        s.run("MERGE (g:Gene {name: 'TREM2'})")
        s.run("MERGE (d:Disease {name: \"Alzheimer's disease\"})")
        s.run("""
            MATCH (g:Gene {name: 'TREM2'}), (d:Disease {name: "Alzheimer's disease"})
            MERGE (g)-[:ASSOCIATED_WITH]->(d)
        """)

        # Query back
        result = s.run("""
            MATCH (n)-[r]->(m)
            RETURN n.name AS from, type(r) AS relation, m.name AS to
        """)
        print("Graph relationships:")
        for row in result:
            print(f"  {row['from']} --[{row['relation']}]--> {row['to']}")

        # Cleanup
        s.run("MATCH (n) DETACH DELETE n")
        print("\nTest data cleaned up.")

    driver.close()
    print("\nNeo4j Aura is ready!")

except Exception as e:
    print(f"\nConnection failed: {e}")
    print("\nCheck:")
    print("  1. URI starts with neo4j+s://")
    print("  2. Username matches the downloaded txt file")
    print("  3. Password is correct — no spaces or quotes")
    print("  4. Aura instance is running (green dot in console)")