from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""

    # NCBI Entrez
    entrez_email: str = ""
    entrez_api_key: str = ""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Pipeline
    pubmed_max_results: int = 60
    hypothesis_max: int = 5

    # Auth + Database
    clerk_secret_key: str = ""
    database_url: str = ""

    # Rate limiting
    free_queries_per_day: int = 3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
