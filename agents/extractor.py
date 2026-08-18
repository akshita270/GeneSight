from __future__ import annotations
import re
import logging
from utils.text import paper_text

logger = logging.getLogger("genesight")

# ── No spaCy — pure regex extraction to stay within Render free memory ──

NON_GENE_CHEMICALS = {
    "iron", "calcium", "zinc", "copper", "magnesium", "sodium", "potassium",
    "glucose", "insulin", "dopamine", "serotonin", "cortisol", "cholesterol",
    "amyloid", "tau", "lipid", "oxygen", "nitrogen", "carbon", "water",
    "n6-methyladenosine", "adenosine", "glutamate", "gaba", "acetylcholine",
    "norepinephrine", "melatonin", "adrenaline", "histamine", "cytokine",
    "interleukin", "fibrin", "collagen", "albumin", "hemoglobin", "ferritin",
    "vitamin", "steroid", "hormone", "enzyme", "antibody", "antigen",
    "mrna", "rrna", "trna", "cdna", "dna", "rna", "atp", "adp", "amp",
    "nadh", "nadph", "camp", "cgmp", "ros", "no", "co2", "h2o2",
}

KNOWN_DISEASES = [
    "alzheimer's disease", "alzheimer disease", "parkinson's disease",
    "parkinson disease", "huntington's disease", "huntington disease",
    "amyotrophic lateral sclerosis", "multiple sclerosis", "epilepsy",
    "schizophrenia", "breast cancer", "lung cancer", "colorectal cancer",
    "colon cancer", "prostate cancer", "ovarian cancer", "pancreatic cancer",
    "liver cancer", "gastric cancer", "bladder cancer", "melanoma",
    "glioblastoma", "glioma", "leukemia", "lymphoma", "myeloma",
    "diabetes", "type 2 diabetes", "type 1 diabetes", "obesity",
    "rheumatoid arthritis", "osteoarthritis", "crohn's disease", "lupus",
    "stroke", "heart failure", "hypertension", "atherosclerosis",
    "cystic fibrosis", "autism", "autism spectrum disorder",
    "depression", "bipolar disorder", "als", "dementia", "myopathy",
    "cardiomyopathy", "neuropathy", "retinopathy", "nephropathy",
    "inflammatory bowel disease", "ulcerative colitis", "psoriasis",
    "asthma", "copd", "fibrosis", "cirrhosis", "hepatitis",
    "covid-19", "sars-cov-2", "hiv", "aids", "tuberculosis", "malaria",
]

# Regex: gene symbols are 2-10 uppercase letters optionally followed by digits/dashes
GENE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]+)?)\b')

# Common gene suffixes/prefixes that help validate
GENE_INDICATORS = re.compile(
    r'\b(gene|mutation|variant|expression|protein|kinase|receptor|'
    r'transcription factor|pathway|signaling|deficiency|knockout|'
    r'overexpression|methylation|phosphorylation|deletion|amplification)\b',
    re.IGNORECASE
)

NON_GENE_WORDS = {
    # Lab / measurement acronyms
    "DNA", "RNA", "PCR", "MRI", "CT", "PET", "NMR", "ELISA", "FACS",
    # Organisations / standards
    "USA", "FDA", "WHO", "NIH", "CDC", "EMA",
    # Small molecules / metabolites (not genes)
    "ATP", "ADP", "AMP", "NAD", "FAD", "GTP", "GDP", "CTP", "UTP",
    "CAMP", "CGMP", "ROS", "NO", "CO",
    # Statistics
    "IC50", "EC50", "LD50", "OR", "HR", "RR", "CI", "SD", "SE",
    # Diseases (abbreviations, not genes)
    "HIV", "AIDS", "TB", "COPD", "IBD", "COVID", "SARS",
    # Two-letter non-genes
    "NF", "AP", "SP", "YY", "IL", "IC", "EC",
}

DISEASE_STOPWORDS = {
    "microglia", "neurons", "cells", "pathway", "signaling", "expression",
    "activity", "function", "response", "mechanism", "study", "analysis",
    "model", "patients", "subjects", "tissue", "brain", "cortex",
}

NON_DISEASE_TERMS = {
    "inflammation", "neuroinflammation", "oxidative stress", "apoptosis",
    "autophagy", "mitophagy", "aggregation", "phosphorylation", "methylation",
    "dysfunction", "degeneration", "toxicity", "clearance", "fibrillation",
    "oligomerization", "neurodegeneration", "excitotoxicity",
}


def extract_genes(text: str) -> set[str]:
    """Extract gene symbols using regex patterns."""
    genes = set()
    for match in GENE_PATTERN.finditer(text):
        symbol = match.group(1)
        # Skip if in known non-gene words
        if symbol in NON_GENE_WORDS:
            continue
        # Skip if lowercase version is a chemical
        if symbol.lower() in NON_GENE_CHEMICALS:
            continue
        # Must be 2-10 chars, start with letter, contain at least one more letter
        if len(symbol) < 2 or len(symbol) > 12:
            continue
        # Prefer symbols with mixed letters+digits (e.g. BRCA1, TP53, LRRK2)
        # or pure uppercase short names (e.g. APOE, CDK5)
        if re.match(r'^[A-Z]{2,}[0-9]*(-[A-Z0-9]+)?$', symbol):
            genes.add(symbol)
    return genes


def extract_diseases(text: str) -> set[str]:
    """Extract disease names using known disease list + pattern matching."""
    diseases = set()
    text_lower = text.lower()

    # Match known diseases first
    for disease in KNOWN_DISEASES:
        if disease in text_lower:
            diseases.add(disease.title().replace("'S", "'s"))

    # Also catch patterns like "X disease", "X disorder", "X syndrome", "X cancer"
    pattern = re.compile(
        r'\b([A-Z][a-z]+(?:\s+[A-Za-z]+){0,3})\s+'
        r'(disease|disorder|syndrome|cancer|carcinoma|tumor|tumour|'
        r'deficiency|insufficiency|failure|injury)\b',
        re.IGNORECASE
    )
    for match in pattern.finditer(text):
        full = match.group(0).strip()
        if full.lower() not in NON_DISEASE_TERMS:
            words = full.split()
            # Filter trailing stopwords
            while words and words[-1].lower() in DISEASE_STOPWORDS:
                words.pop()
            cleaned = " ".join(words).strip()
            if cleaned and len(cleaned) > 4:
                diseases.add(cleaned.title())

    return diseases


class ExtractionAgent:
    def __init__(self):
        pass

    async def run(self, papers):
        genes: set[str] = set()
        diseases: set[str] = set()

        for p in papers:
            text = paper_text(p)
            genes.update(extract_genes(text))
            diseases.update(extract_diseases(text))

        # Remove genes that are actually disease names
        disease_words = {w for d in diseases for w in d.upper().split()}
        genes = {g for g in genes if g not in disease_words}

        # Limit to top hits to keep downstream processing fast
        genes = set(list(genes)[:30])
        diseases = set(list(diseases)[:20])

        logger.info("Extracted %d genes, %d diseases", len(genes), len(diseases))

        return {
            "genes":    list(genes),
            "diseases": list(diseases),
            "proteins": [],
        }
