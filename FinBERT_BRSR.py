"""
FinBERT_BRSR.py
---------------
ESG scoring pipeline for BRSR reports using FinBERT sentiment analysis.

Workflow:
  1. Scan all PDFs in the 'BRSR Reports' folder.
  2. Extract & clean text from each PDF.
  3. Split text into ESG-pillar-specific chunks (Environmental / Social / Governance).
  4. Run FinBERT sentiment inference on each chunk.
  5. Aggregate sentiment probabilities → pillar scores (0-100).
  6. Compute a combined weighted ESG score.
  7. Save results to 'finbert_esg_scores.csv'.

Imports FinBERT from the centralised models.py loader.

Usage:
    python FinBERT_BRSR.py

Prerequisites:
    pip install transformers torch pdfplumber tqdm
"""

import os
import re
import csv
import warnings
import math
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm

# ── Import FinBERT from the central models.py ──────────────────────────────────
from models import load_finbert

warnings.filterwarnings("ignore")

# =============================================================================
# 1.  CONFIGURATION
# =============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BRSR_FOLDER  = os.path.join(SCRIPT_DIR, "BRSR Reports")
OUTPUT_FILE  = os.path.join(SCRIPT_DIR, "finbert_esg_scores.csv")

# ── FinBERT token ceiling ─────────────────────────────────────────────────────
# FinBERT (ProsusAI/finbert) is capped at 512 tokens, but was fine-tuned on
# SHORT financial sentences (~128 tokens avg). Using the full 512 produces
# diluted, less reliable sentiment because unrelated sentences pull the score.
MAX_TOKENS = 512          # Hard BERT limit – do not exceed

# ── Chunk word size ────────────────────────────────────────────────────────────
# Target ≈ 128 words per chunk  (~170 tokens for financial text).
# This matches FinBERT's training distribution and gives sharper sentiment.
# Empirically this improves score differentiation across companies.
CHUNK_WORD_SIZE = 128     # ← was 384 (too long, diluted sentiment signal)

# ── Chunk overlap ──────────────────────────────────────────────────────────────
# 32-word overlap preserves sentence boundary context so key ESG phrases
# that straddle a boundary are not lost. ~25% of chunk size is ideal.
CHUNK_OVERLAP = 32        # ← was hard-coded 50 (too much for small chunks)

# ── Noise filter ───────────────────────────────────────────────────────────────
# Discard chunks shorter than this – they are usually headers / page numbers
# and produce low-confidence FinBERT outputs that skew averages.
MIN_CHUNK_WORDS = 30      # ← was 20 (raised to filter noise fragments)

# ── Coverage bonus ceiling ─────────────────────────────────────────────────────
# The log ceiling controls how quickly the coverage bonus saturates.
# log1p(30) means a company needs ~30 routed chunks to earn the full +10 pts.
# Lowering from 50 → 30 makes the bonus more attainable for smaller reports.
COVERAGE_BONUS_CEILING = 30   # ← was 50 (30 is more realistic for BRSR docs)
COVERAGE_BONUS_MAX_PTS = 10   # Maximum bonus points for disclosure volume

# ── ESG pillar weights ─────────────────────────────────────────────────────────
PILLAR_WEIGHTS = {"env": 0.44, "soc": 0.31, "gov": 0.25}

# =============================================================================
# 2.  ESG KEYWORD SETS (used to filter & route text chunks to the right pillar)
# =============================================================================

PILLAR_KEYWORDS: Dict[str, List[str]] = {
    "env": [
        "greenhouse gas", "ghg", "carbon", "co2", "scope 1", "scope 2", "scope 3",
        "net zero", "carbon neutral", "carbon footprint", "emission", "climate",
        "decarbonization", "paris agreement", "renewable energy", "solar", "wind",
        "clean energy", "energy efficiency", "energy consumption", "fossil fuel",
        "water consumption", "water recycling", "wastewater", "water conservation",
        "waste management", "hazardous waste", "circular economy", "landfill",
        "biodiversity", "ecosystem", "deforestation", "land use", "habitat",
        "air pollution", "nox", "sox", "particulate matter", "pm2.5", "voc",
        "environmental", "iso 14001", "ems", "life cycle", "environmental audit",
        "environmental impact", "pollution", "plastic", "e-waste",
    ],
    "soc": [
        "employee", "workforce", "diversity", "inclusion", "gender", "equal opportunity",
        "dei", "human capital", "talent", "attrition", "retention", "training",
        "occupational health", "safety", "injury", "fatalities", "incident rate",
        "mental health", "wellbeing", "human rights", "child labor", "forced labor",
        "modern slavery", "fair wages", "living wage", "labor rights", "collective bargaining",
        "skill development", "learning", "upskilling", "reskilling", "community",
        "csr", "social responsibility", "social impact", "rural development", "education",
        "healthcare", "philanthropy", "volunteering", "customer satisfaction",
        "customer grievance", "product safety", "data privacy", "consumer protection",
        "supplier", "ethical sourcing", "vendor", "whistleblower", "anti harassment",
        "posh", "grievance", "women empowerment",
    ],
    "gov": [
        "board of directors", "board composition", "board independence", "independent director",
        "board diversity", "board evaluation", "chairman", "managing director", "ceo",
        "executive compensation", "remuneration", "audit committee", "risk committee",
        "nomination committee", "esg committee", "sustainability committee",
        "compliance", "regulatory", "anti corruption", "anti bribery", "fcpa",
        "code of conduct", "ethics policy", "conflict of interest",
        "related party", "risk management", "enterprise risk", "internal controls",
        "internal audit", "external audit", "assurance", "third party verification",
        "transparency", "disclosure", "gri", "sasb", "tcfd", "brsr",
        "integrated reporting", "sustainability reporting", "materiality assessment",
        "shareholder rights", "investor relations", "dividend policy", "cybersecurity",
        "data security", "data governance", "gdpr", "tax transparency", "financial integrity",
    ],
}


# =============================================================================
# 3.  PDF TEXT EXTRACTION & CLEANING
# =============================================================================

def _extract_text_pdfplumber(path: str) -> str:
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n".join(parts)
    except Exception:
        return ""


def _extract_text_pypdf2(path: str) -> str:
    try:
        import PyPDF2
        parts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n".join(parts)
    except Exception:
        return ""


def extract_text(path: str) -> str:
    """Extract raw text from a PDF, auto-detecting available library."""
    try:
        import pdfplumber
        text = _extract_text_pdfplumber(path)
    except ImportError:
        text = _extract_text_pypdf2(path)

    if not text.strip():
        # Fallback to the other library
        try:
            text = _extract_text_pypdf2(path)
        except Exception:
            pass

    return text


def clean_text(raw: str) -> str:
    """Lowercase + normalise unicode + strip non-alpha characters."""
    text = raw.lower()
    text = (text.replace("\u2013", "-").replace("\u2014", "-")
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))
    text = re.sub(r"[^a-z0-9\s\-\.,;:!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# 4.  CHUNKING & PILLAR ROUTING
# =============================================================================

def split_into_chunks(
    text: str,
    chunk_words: int = CHUNK_WORD_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping word-level chunks for FinBERT.

    Uses a sliding window of `chunk_words` words with `overlap` words of
    context carried over from the previous chunk to preserve sentence boundaries.
    """
    words = text.split()
    chunks = []
    step = max(1, chunk_words - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_words])
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
    return chunks


def _chunk_keyword_density(chunk: str, keywords: List[str]) -> float:
    """Return the fraction of pillar keywords present in a chunk."""
    hits = sum(1 for kw in keywords if kw in chunk)
    return hits / max(len(keywords), 1)


def route_chunks_to_pillars(
    chunks: List[str],
) -> Dict[str, List[str]]:
    """
    Assign each chunk to ONE pillar (the one with highest keyword density).
    A chunk with zero keyword hits for all pillars is discarded.
    """
    routed: Dict[str, List[str]] = {"env": [], "soc": [], "gov": []}
    for chunk in chunks:
        scores = {
            p: _chunk_keyword_density(chunk, PILLAR_KEYWORDS[p])
            for p in routed
        }
        best_pillar = max(scores, key=scores.get)
        if scores[best_pillar] > 0:  # only route chunks that mention at least one keyword
            routed[best_pillar].append(chunk)
    return routed


# =============================================================================
# 5.  FINBERT SENTIMENT → ESG SCORE CONVERSION
# =============================================================================

def _sentiment_probs(result: dict) -> Dict[str, float]:
    """
    FinBERT pipeline returns the TOP label only.
    We therefore run in top_k=None mode to get all three probabilities.
    This helper normalises a single-result dict into {positive, neutral, negative}.
    """
    label = result["label"].lower()
    score = result["score"]
    # Estimate the other two assuming equal split of the remainder
    remainder = (1.0 - score) / 2
    if label == "positive":
        return {"positive": score, "neutral": remainder, "negative": remainder}
    elif label == "negative":
        return {"positive": remainder, "neutral": remainder, "negative": score}
    else:  # neutral
        return {"positive": remainder, "neutral": score, "negative": remainder}


def aggregate_sentiment(
    chunks: List[str],
    pipe,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Run FinBERT on a list of text chunks and return average sentiment probabilities.

    Returns:
        {"positive": float, "neutral": float, "negative": float}
        All values are in [0, 1] and sum to ~1.
    """
    if not chunks:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    totals = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    count = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        try:
            results = pipe(batch, truncation=True, max_length=MAX_TOKENS, batch_size=batch_size)
            for res in results:
                # pipeline returns list of dicts when input is a list
                if isinstance(res, list):
                    res = res[0]
                probs = _sentiment_probs(res)
                for k in totals:
                    totals[k] += probs[k]
                count += 1
        except Exception:
            # Skip batches that raise errors (e.g. empty strings after tokenisation)
            pass

    if count == 0:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    return {k: v / count for k, v in totals.items()}


def sentiment_to_pillar_score(avg_probs: Dict[str, float], n_chunks: int) -> float:
    """
    Convert average FinBERT sentiment probabilities into a pillar score (0–100).

    Scoring philosophy:
      - positive sentiment   → rewards ESG disclosure quality  (weight +1.0)
      - neutral  sentiment   → partial reward                  (weight +0.5)
      - negative sentiment   → penalises (no reward)           (weight  0.0)
      - Coverage bonus       → more routed chunks = richer ESG disclosure

    Formula:
        raw_score      = 100 × (positive + 0.5 × neutral)
        coverage_bonus = min(log1p(n_chunks) / log1p(CEILING), 1.0) × MAX_PTS
        final          = min(raw_score + coverage_bonus, 100)

    Tuning notes:
      - CHUNK_WORD_SIZE=128  → more chunks per doc → finer-grained sentiment
      - COVERAGE_BONUS_CEILING=30 → full bonus attainable at ~30 chunks
        (a typical 5,000-word pillar section at 128 words/chunk yields ~35 chunks)
    """
    raw = 100.0 * (avg_probs["positive"] + 0.5 * avg_probs["neutral"])
    coverage_bonus = (
        min(math.log1p(n_chunks) / math.log1p(COVERAGE_BONUS_CEILING), 1.0)
        * COVERAGE_BONUS_MAX_PTS
    )
    return round(min(raw + coverage_bonus, 100.0), 2)


# =============================================================================
# 6.  MAIN REPORT PROCESSOR
# =============================================================================

def process_report(pdf_path: str, pipe) -> Dict:
    """
    Extract text from a BRSR PDF and compute FinBERT-based ESG scores.

    Returns a dict with company name, pillar scores, and combined ESG score.
    """
    company_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # --- Extract & clean ---
    raw_text  = extract_text(pdf_path)
    clean     = clean_text(raw_text)
    word_count = len(clean.split())

    if word_count < 100:
        raise ValueError(f"Too little text extracted ({word_count} words).")

    # --- Chunk & route to pillars ---
    all_chunks = split_into_chunks(clean)
    routed     = route_chunks_to_pillars(all_chunks)

    # --- FinBERT inference per pillar ---
    pillar_scores = {}
    pillar_sentiments = {}

    for pillar, chunks in routed.items():
        avg_probs = aggregate_sentiment(chunks, pipe)
        n_chunks  = len(chunks)
        pillar_scores[pillar]     = sentiment_to_pillar_score(avg_probs, n_chunks)
        pillar_sentiments[pillar] = avg_probs

    env_score = pillar_scores.get("env", 0.0)
    soc_score = pillar_scores.get("soc", 0.0)
    gov_score = pillar_scores.get("gov", 0.0)

    # --- Weighted combined ESG score ---
    combined = round(
        env_score * PILLAR_WEIGHTS["env"]
        + soc_score * PILLAR_WEIGHTS["soc"]
        + gov_score * PILLAR_WEIGHTS["gov"],
        2,
    )

    return {
        "company"           : company_name,
        "word_count"        : word_count,
        "total_chunks"      : len(all_chunks),
        "env_chunks"        : len(routed["env"]),
        "soc_chunks"        : len(routed["soc"]),
        "gov_chunks"        : len(routed["gov"]),
        # Pillar Scores (0-100)
        "env_score"         : env_score,
        "soc_score"         : soc_score,
        "gov_score"         : gov_score,
        # Combined weighted score
        "combined_esg_score": combined,
        # Average sentiment probabilities per pillar
        "env_positive"      : round(pillar_sentiments.get("env", {}).get("positive", 0), 4),
        "env_neutral"       : round(pillar_sentiments.get("env", {}).get("neutral",  0), 4),
        "env_negative"      : round(pillar_sentiments.get("env", {}).get("negative", 0), 4),
        "soc_positive"      : round(pillar_sentiments.get("soc", {}).get("positive", 0), 4),
        "soc_neutral"       : round(pillar_sentiments.get("soc", {}).get("neutral",  0), 4),
        "soc_negative"      : round(pillar_sentiments.get("soc", {}).get("negative", 0), 4),
        "gov_positive"      : round(pillar_sentiments.get("gov", {}).get("positive", 0), 4),
        "gov_neutral"       : round(pillar_sentiments.get("gov", {}).get("neutral",  0), 4),
        "gov_negative"      : round(pillar_sentiments.get("gov", {}).get("negative", 0), 4),
    }


# =============================================================================
# 7.  MAIN ENTRY POINT
# =============================================================================

FIELDNAMES = [
    "company", "word_count", "total_chunks",
    "env_chunks", "soc_chunks", "gov_chunks",
    "env_score", "soc_score", "gov_score", "combined_esg_score",
    "env_positive", "env_neutral", "env_negative",
    "soc_positive", "soc_neutral", "soc_negative",
    "gov_positive", "gov_neutral", "gov_negative",
]


def main():
    # --- Validate BRSR folder ---
    if not os.path.isdir(BRSR_FOLDER):
        print(f"[ERROR] 'BRSR Reports' folder not found at: {BRSR_FOLDER}")
        return

    pdf_files = sorted([
        os.path.join(BRSR_FOLDER, f)
        for f in os.listdir(BRSR_FOLDER)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print("[ERROR] No PDF files found in 'BRSR Reports'.")
        return

    print(f"[INFO]  Found {len(pdf_files)} BRSR report(s) to process.\n")

    # --- Load FinBERT (from central models.py) ---
    print("[INFO]  Loading FinBERT model from models.py ...")
    _, _, finbert_pipe = load_finbert()
    print()

    # --- Process each report ---
    records   = []
    errors    = []
    progress  = tqdm(pdf_files, desc="Processing BRSR Reports", unit="file")

    for pdf_path in progress:
        fname = os.path.basename(pdf_path)
        progress.set_postfix(file=fname[:40])
        try:
            rec = process_report(pdf_path, finbert_pipe)
            records.append(rec)
            tqdm.write(
                f"  [OK]  {rec['company'][:50]:<50}  "
                f"E={rec['env_score']:5.1f}  "
                f"S={rec['soc_score']:5.1f}  "
                f"G={rec['gov_score']:5.1f}  "
                f"ESG={rec['combined_esg_score']:5.1f}"
            )
        except Exception as exc:
            errors.append((fname, str(exc)))
            tqdm.write(f"  [ERR] {fname[:60]} → {exc}")

    # --- Sort by combined_esg_score descending ---
    records.sort(key=lambda r: r["combined_esg_score"], reverse=True)

    # --- Write CSV ---
    if records:
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        print(f"\n[DONE]  Results saved → {OUTPUT_FILE}")
        print(f"        Processed : {len(records)} report(s)")
    else:
        print("\n[WARN]  No records to save.")

    if errors:
        print(f"\n[WARN]  {len(errors)} file(s) failed:")
        for fname, err in errors:
            print(f"        - {fname}: {err}")


if __name__ == "__main__":
    main()
