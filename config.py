from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # NCBI Entrez
    entrez_email: str = os.getenv("ENTREZ_EMAIL", "")
    entrez_api_key: str = os.getenv("ENTREZ_API_KEY", "")

    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")

    # Pipeline settings
    pubmed_max_results: int = int(os.getenv("PUBMED_MAX_RESULTS", "60"))
    hypothesis_max: int = int(os.getenv("HYPOTHESIS_MAX", "5"))

    # Auth + Database
    clerk_secret_key: str = os.getenv("CLERK_SECRET_KEY", "")
    database_url: str = os.getenv("DATABASE_URL", "")

    # Rate limit: free queries per user per day
    free_queries_per_day: int = int(os.getenv("FREE_QUERIES_PER_DAY", "3"))

settings = Settings()