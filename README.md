# GeneSight · Genomics Hypothesis Engine

An agentic AI system that automatically generates scientific hypotheses from genomics literature and biological databases using an 8-agent pipeline.

## Architecture

```
User Query → FastAPI Backend → 8-Agent Pipeline → Results
                                      │
          ┌───────────────────────────┤
          │  1. Task Planner          │ GPT-4o
          │  2. Literature Retrieval  │ PubMed / NCBI Entrez
          │  3. Info Extraction       │ spaCy biomedical NER (bc5cdr)
          │  4. Genomics DB           │ NCBI Gene + UniProt
          │  5. Knowledge Graph       │ Neo4j
          │  6. Hypothesis Generation │ GPT-4o
          │  7. Evidence Validation   │ keyword scoring
          │  8. Report Generation     │ GPT-4o
          └───────────────────────────┘
```

## Prerequisites

- Python 3.11
- A running Neo4j instance (local or [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/))
- OpenAI API key (GPT-4o access)
- NCBI Entrez API key (free at https://www.ncbi.nlm.nih.gov/account/)

## Local Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd "genomic hypothesis"

# 2. Create and activate a virtual environment
python3.11 -m venv env311
source env311/bin/activate        # Windows: env311\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the biomedical NER model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

# 5. Copy and fill in environment variables
cp .env.example .env
# Edit .env with your keys (see Environment Variables section below)

# 6. Start the FastAPI backend
uvicorn main:app --host 0.0.0.0 --port 8000

# 7. In a separate terminal, start the Streamlit UI
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Environment Variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=sk-...

# NCBI Entrez (required for PubMed search)
ENTREZ_EMAIL=your@email.com
ENTREZ_API_KEY=your_ncbi_api_key

# Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Optional tuning
PUBMED_MAX_RESULTS=20   # papers to retrieve per query
HYPOTHESIS_MAX=5        # hypotheses to generate

# Streamlit → FastAPI URL (set this when running separately)
API_URL=http://localhost:8000
```

## Deployment on Render

The `render.yaml` configures the FastAPI backend as a web service.

1. Push this repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com) and connect the repo
3. Render will auto-detect `render.yaml` and configure the build/start commands
4. Set the following secret environment variables in the Render dashboard:
   - `OPENAI_API_KEY`
   - `ENTREZ_EMAIL`
   - `ENTREZ_API_KEY`
   - `NEO4J_URI`
   - `NEO4J_USER`
   - `NEO4J_PASSWORD`
5. For the Streamlit UI, deploy a second Render service (or use [Streamlit Community Cloud](https://streamlit.io/cloud)) and set `API_URL` to the FastAPI service URL

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Start a pipeline job. Body: `{"query": "..."}` |
| `GET` | `/status/{job_id}` | Poll job status: `queued \| running \| done \| error` |
| `GET` | `/result/{job_id}` | Fetch completed results |
| `GET` | `/health` | Health check |

Interactive API docs: http://localhost:8000/docs

## Data Sources

| Source | Purpose |
|--------|---------|
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/) | Biomedical literature retrieval |
| [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/) | Gene metadata & IDs |
| [UniProt](https://www.uniprot.org/) | Protein function & IDs |
| [Neo4j](https://neo4j.com/) | Knowledge graph persistence |

## Example Query

> "Identify potential new gene relationships related to Alzheimer's disease"

**Output includes:**
- List of relevant PubMed papers
- Knowledge graph of gene–disease relationships
- Ranked hypotheses with confidence scores and supporting evidence
