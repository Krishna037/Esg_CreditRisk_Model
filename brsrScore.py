import os
import re
import math
import csv
import warnings

# ── PDF library auto-detect ──────────────────────────────────────────────────
try:
    import pdfplumber
    PDF_LIB = "pdfplumber"
except ImportError:
    try:
        import PyPDF2
        PDF_LIB = "PyPDF2"
    except ImportError:
        raise ImportError(
            "Install a PDF library first:\n"
            "  pip install pdfplumber\n"
            "  OR: pip install PyPDF2"
        )

warnings.filterwarnings("ignore")

# =============================================================================
# 1.  OPTIMIZED ABSOLUTE SCORING CEILINGS (Lowered to boost scores)
#     These define the threshold for a "100" score.
# =============================================================================

# Mentions per 1000 words that earns a density score of 100
DENSITY_CEILING = {
    "env": 8.5,    # Lowered from 12.0 (More achievable for Env topics)
    "soc": 4.5,    # Lowered from 6.0  (More achievable for Social topics)
    "gov": 8.5,    # Lowered from 12.0 (More achievable for Gov topics)
}

# Raw keyword count that earns a volume score of 100
VOLUME_CEILING = {
    "env": 500,    # Lowered from 800
    "soc": 350,    # Lowered from 500
    "gov": 600,    # Lowered from 1000
}

# =============================================================================
# 2.  KEYWORD LISTS
# =============================================================================

environmental_keywords = [
    "greenhouse gas", "ghg emissions", "carbon emissions", "co2 emissions",
    "scope 1", "scope 2", "scope 3", "net zero", "carbon neutral",
    "carbon footprint", "carbon offset", "emission intensity",
    "climate change", "climate risk", "climate strategy", "decarbonization",
    "paris agreement", "ndc", "transition risk", "energy consumption", 
    "energy efficiency", "renewable energy", "solar energy", "wind energy", 
    "clean energy", "energy intensity", "energy management", "fossil fuel", 
    "fuel consumption", "water consumption", "water intensity", "water recycling",
    "water withdrawal", "water discharge", "wastewater treatment",
    "water conservation", "water stress", "freshwater", "waste management", 
    "waste generated", "hazardous waste", "non-hazardous waste", "waste recycled", 
    "waste disposed", "circular economy", "zero waste", "landfill", "e-waste",
    "plastic waste", "biodegradable", "biodiversity", "ecosystem", "deforestation", 
    "land use", "habitat conservation", "protected areas", "air pollution", "nox", 
    "sox", "particulate matter", "pm2.5", "volatile organic compounds", "voc", 
    "air quality", "environmental management", "iso 14001", "ems", 
    "life cycle assessment", "environmental compliance", "environmental audit",
    "environmental impact assessment", "eia",
]

social_keywords = [
    "employee engagement", "employee satisfaction", "workforce diversity",
    "gender diversity", "equal opportunity", "inclusion", "dei",
    "diversity equity inclusion", "human capital", "talent management",
    "employee retention", "attrition rate", "workforce participation",
    "occupational health", "workplace safety", "lost time injury",
    "ltifr", "fatalities", "incident rate", "safety training",
    "health benefits", "mental health", "employee wellbeing",
    "ergonomics", "industrial safety", "human rights policy", "child labor", 
    "forced labor", "modern slavery", "fair wages", "living wage", "labor rights", 
    "freedom of association", "collective bargaining", "skill development", 
    "employee training", "learning and development", "upskilling", "reskilling", 
    "leadership development", "career progression", "community development", 
    "csr initiatives", "corporate social responsibility", "social impact", 
    "rural development", "education initiatives", "healthcare initiatives", 
    "community investment", "philanthropy", "volunteering", "stakeholder engagement",
    "customer satisfaction", "customer grievance", "product safety",
    "data privacy", "consumer protection", "responsible marketing",
    "product quality", "customer trust", "supplier code of conduct", 
    "supplier diversity", "ethical sourcing", "supply chain labor standards", 
    "vendor audits", "accessibility", "inclusive design", 
    "differently abled inclusion", "women empowerment", "minority inclusion",
    "whistleblower mechanism", "anti harassment policy", "posh compliance",
    "grievance redressal", "ethics training",
]

governance_keywords = [
    "board of directors", "board composition", "board independence",
    "independent director", "board diversity", "board effectiveness",
    "board evaluation", "chairman", "managing director", "ceo",
    "executive compensation", "remuneration policy", "audit committee", 
    "risk committee", "nomination committee", "remuneration committee", 
    "esg committee", "sustainability committee", "stakeholder committee",
    "compliance", "regulatory compliance", "legal compliance",
    "anti corruption", "anti bribery", "fcpa", "uk bribery act",
    "code of conduct", "ethics policy", "conflict of interest",
    "related party transactions", "risk management", "enterprise risk", 
    "risk framework", "internal controls", "material risk", "risk appetite",
    "business continuity", "crisis management", "internal audit", 
    "external audit", "statutory audit", "assurance", "third party verification", 
    "limited assurance", "reasonable assurance", "transparency", "disclosure", 
    "gri", "sasb", "tcfd", "brsr", "integrated reporting", "sustainability reporting", 
    "annual report", "materiality assessment", "double materiality",
    "shareholder rights", "investor relations", "minority shareholders",
    "voting rights", "dividend policy", "say on pay", "cybersecurity", 
    "data security", "information security", "data governance", "privacy policy", 
    "gdpr", "tax transparency", "tax governance", "fair tax", "financial integrity",
]

PILLAR_KW_COUNT = {
    "env": len(environmental_keywords),
    "soc": len(social_keywords),
    "gov": len(governance_keywords),
}

# =============================================================================
# 3.  EXTRACTION & UTILS
# =============================================================================

def extract_text(path):
    parts = []
    if PDF_LIB == "pdfplumber":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: parts.append(t)
    else:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t: parts.append(t)
    return "\n".join(parts)

def clean_text(raw):
    text = raw.lower()
    text = (text.replace("\u2013", "-").replace("\u2014", "-")
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"'))
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def count_keywords(text, keywords):
    counts = {}
    for kw in sorted(keywords, key=len, reverse=True):
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        counts[kw] = len(re.findall(pattern, text))
    return counts

# =============================================================================
# 4.  SCORING LOGIC
# =============================================================================

def density_score(raw_count, word_count, pillar):
    if word_count < 1: return 0.0
    d = (raw_count / word_count) * 1000
    return min(d / DENSITY_CEILING[pillar] * 100, 100.0)

def coverage_score(counts, pillar):
    total = PILLAR_KW_COUNT[pillar]
    if total == 0: return 0.0
    seen = sum(1 for v in counts.values() if v > 0)
    return (seen / total) * 100.0

def volume_score(raw_count, pillar):
    ceiling_log = math.log1p(VOLUME_CEILING[pillar])
    if ceiling_log == 0: return 0.0
    return min(math.log1p(raw_count) / ceiling_log * 100, 100.0)

def pillar_score(raw_count, word_count, counts, pillar):
    # Weight within each pillar: 50% density, 30% coverage, 20% volume
    d = density_score(raw_count, word_count, pillar)
    c = coverage_score(counts, pillar)
    v = volume_score(raw_count, pillar)
    return round(0.50 * d + 0.30 * c + 0.20 * v, 2)

# =============================================================================
# 5.  PROCESSING & CALCULATION
# =============================================================================

def process_report(pdf_path):
    name = os.path.splitext(os.path.basename(pdf_path))[0]
    raw_text = extract_text(pdf_path)
    clean    = clean_text(raw_text)
    wc       = max(len(clean.split()), 1)

    env_counts = count_keywords(clean, environmental_keywords)
    soc_counts = count_keywords(clean, social_keywords)
    gov_counts = count_keywords(clean, governance_keywords)

    env_raw = int(sum(env_counts.values()))
    soc_raw = int(sum(soc_counts.values()))
    gov_raw = int(sum(gov_counts.values()))

    # Calculate pillar scores (0-100)
    env_s = pillar_score(env_raw, wc, env_counts, "env")
    soc_s = pillar_score(soc_raw, wc, soc_counts, "soc")
    gov_s = pillar_score(gov_raw, wc, gov_counts, "gov")

    # ── WEIGHTED CALCULATION: E=0.44, S=0.31, G=0.26 ──
    combined = round((env_s * 0.44) + (soc_s * 0.31) + (gov_s * 0.26), 2)

    return {
        "company"              : name,
        "word_count"           : wc,
        "env_raw"              : env_raw,
        "soc_raw"              : soc_raw,
        "gov_raw"              : gov_raw,
        "env_score"            : env_s,
        "soc_score"            : soc_s,
        "gov_score"            : gov_s,
        "combined_esg_score"   : combined,
        "top_env_keywords"     : ", ".join(sorted(env_counts, key=env_counts.get, reverse=True)[:5]),
        "top_soc_keywords"     : ", ".join(sorted(soc_counts, key=soc_counts.get, reverse=True)[:5]),
        "top_gov_keywords"     : ", ".join(sorted(gov_counts, key=gov_counts.get, reverse=True)[:5]),
    }

# =============================================================================
# 6.  MAIN EXECUTION
# =============================================================================

FIELDNAMES = [
    "company", "word_count", "env_raw", "soc_raw", "gov_raw",
    "env_score", "soc_score", "gov_score", "combined_esg_score",
    "top_env_keywords", "top_soc_keywords", "top_gov_keywords"
]

def main():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    brsr_folder = os.path.join(script_dir, "BRSR Reports")

    if not os.path.isdir(brsr_folder):
        print(f"Error: Folder 'BRSR Reports' not found in {script_dir}")
        return

    pdf_files = [os.path.join(brsr_folder, f) for f in os.listdir(brsr_folder) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("No PDFs found in the folder.")
        return

    print(f"Starting analysis on {len(pdf_files)} reports...")
    records = []
    for path in pdf_files:
        try:
            rec = process_report(path)
            records.append(rec)
            print(f" [OK] {rec['company']:<25} Score: {rec['combined_esg_score']}")
        except Exception as e:
            print(f" [ERR] Skipping {os.path.basename(path)}: {e}")

    if records:
        records.sort(key=lambda r: r["combined_esg_score"], reverse=True)
        out_file = os.path.join(script_dir, "brsr_esg_scores_optimized.csv")
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        print(f"\nCompleted! Results saved to {out_file}")

if __name__ == "__main__":
    main()