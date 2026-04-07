from __future__ import annotations
import spacy
import re

NON_GENE_CHEMICALS = {
    "iron", "calcium", "zinc", "copper", "magnesium", "sodium", "potassium",
    "glucose", "insulin", "dopamine", "serotonin", "cortisol", "cholesterol",
    "amyloid", "tau", "lipid", "oxygen", "nitrogen", "carbon", "water",
    "n6-methyladenosine", "adenosine", "glutamate", "gaba", "acetylcholine",
    "norepinephrine", "melatonin", "adrenaline", "histamine", "cytokine",
    "interleukin", "fibrin", "collagen", "albumin", "hemoglobin", "ferritin",
    "vitamin", "steroid", "hormone", "enzyme", "antibody", "antigen",
}

NON_DISEASE_TERMS = {
    "inflammation", "neuroinflammation", "oxidative stress", "apoptosis",
    "autophagy", "mitophagy", "aggregation", "phosphorylation", "methylation",
    "dysfunction", "degeneration", "toxicity", "clearance",
    "fibrillation", "oligomerization", "neurodegeneration", "excitotoxicity",
}

DISEASE_STOPWORDS = {
    "microglia", "neurons", "cells", "pathway", "signaling", "expression",
    "activity", "function", "response", "mechanism", "study", "analysis",
    "model", "patients", "subjects", "tissue", "brain", "cortex", "hippocampus",
}


def clean_disease(text):
    text = text.strip()
    KNOWN_DISEASES = [
        "alzheimer's disease", "parkinson's disease", "huntington's disease",
        "amyotrophic lateral sclerosis", "multiple sclerosis", "epilepsy",
        "schizophrenia", "breast cancer", "lung cancer", "colorectal cancer",
        "ovarian cancer", "leukemia", "lymphoma", "diabetes", "type 2 diabetes",
        "rheumatoid arthritis", "crohn's disease", "lupus", "stroke",
        "heart failure", "hypertension", "cystic fibrosis", "autism",
        "depression", "bipolar disorder", "als", "dementia",
    ]
    for d in KNOWN_DISEASES:
        if d in text.lower():
            return d.title().replace("'S", "'s").replace(" Als", " ALS")
    words = text.split()
    while words and words[-1].lower() in DISEASE_STOPWORDS:
        words.pop()
    while words and words[0].lower() in {"idiopathic", "sporadic", "familial", "late-onset", "early-onset"}:
        words.pop(0)
    cleaned = " ".join(words).strip()
    return cleaned if cleaned else text


def is_valid_gene(text):
    text = text.strip()
    if text.lower() in NON_GENE_CHEMICALS:
        return False
    if len(text.split()) > 1:
        return False
    if len(text) < 2 or len(text) > 12:
        return False
    if not re.search(r'[A-Za-z]', text):
        return False
    if not re.search(r'[A-Z]', text):
        return False
    if re.match(r'^\d+$', text):
        return False
    if not re.match(r'^[A-Za-z0-9\-]+$', text):
        return False
    return True


# Load model once at startup
print("Loading spaCy biomedical NER model...")
_NLP = spacy.load("en_ner_bc5cdr_md")
print("spaCy model loaded.")


class ExtractionAgent:
    def __init__(self):
        self.nlp = _NLP

    async def run(self, papers):
        genes, diseases, proteins = set(), set(), set()

        for p in papers:
            text = p.get("title", "") + " " + p.get("abstract", "")
            doc = self.nlp(text)

            for ent in doc.ents:
                val = ent.text.strip()

                if ent.label_ == "CHEMICAL":
                    if is_valid_gene(val):
                        genes.add(val)

                elif ent.label_ == "DISEASE":
                    if val.lower() in NON_DISEASE_TERMS:
                        continue
                    if len(val.split()) < 2 and val.lower() not in {"als", "hiv", "ad", "pd"}:
                        continue
                    cleaned = clean_disease(val)
                    if cleaned:
                        diseases.add(cleaned)

        genes = {g for g in genes if g not in diseases}

        print(f"Extracted genes:    {genes}")
        print(f"Extracted diseases: {diseases}")

        return {
            "genes":    list(genes),
            "diseases": list(diseases),
            "proteins": list(proteins),
        }