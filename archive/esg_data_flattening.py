"""
ESG Data Flattening & Scoring Script
=====================================
Transforms raw JSON article data into a structured master dataset
and calculates time-decayed ESG risk scores.

Author: Capstone Project
Date: February 2026
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import warnings
import re

warnings.filterwarnings('ignore')

import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Configuration
ARTICLES_DIR = Path(__file__).parent / "articles"
OUTPUT_FILE = Path(__file__).parent / "incidents_flat.csv"
COMPANY_SUMMARY_FILE = Path(__file__).parent / "company_esg_scores.csv"
WALK_DATA_EXCEL_FILE = Path(__file__).parent / "walk_data_esg_score.xlsx"
ESG_CATEGORIES = ["Environmental", "Social", "Governance"]
TODAY = datetime.now()

# ESG WEIGHTING SCHEME
ESG_WEIGHTS = {
    "Environmental": 0.44,
    "Social": 0.31,
    "Governance": 0.26
}

# Core ESG trigger words to boost relevance (VADER handles the rest)
ESG_IMPLICATION_TRIGGERS = {
    "fraud", "lawsuit", "penalty", "fine", "strike", "violation", "investigation", 
    "scandal", "breach", "boycott", "protest", "corruption", "laundering", "spill", 
    "emission", "toxic", "litigation", "harassment", "allegation"
}


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse various date formats into datetime objects.
    
    Handles formats:
    - "Wed, 25 Jan 2023 08:00:00 GMT" (RSS format)
    - "2026-01-24" (ISO format from parsed_date)
    - Various other common formats
    """
    if date_str is None or pd.isna(date_str) or str(date_str).strip() == '':
        return None
    
    date_str = str(date_str).strip()
    
    # List of date formats to try
    date_formats = [
        "%a, %d %b %Y %H:%M:%S %Z",  # RSS format: Wed, 25 Jan 2023 08:00:00 GMT
        "%a, %d %b %Y %H:%M:%S %z",  # RSS with timezone offset
        "%Y-%m-%d",                   # ISO format: 2026-01-24
        "%Y-%m-%dT%H:%M:%S.%f",      # ISO with microseconds
        "%Y-%m-%dT%H:%M:%S",         # ISO without microseconds
        "%d/%m/%Y",                   # DD/MM/YYYY
        "%m/%d/%Y",                   # MM/DD/YYYY
        "%d-%m-%Y",                   # DD-MM-YYYY
        "%B %d, %Y",                  # January 24, 2026
        "%b %d, %Y",                  # Jan 24, 2026
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Last resort: try pandas parsing
    try:
        return pd.to_datetime(date_str).to_pydatetime()
    except:
        return None


def extract_article_data(json_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract relevant fields from a single JSON article file.
    Handles missing fields gracefully.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract fields with defaults for missing values
        article = {
            'company': data.get('company', None),
            'category': data.get('category', None),
            'date': data.get('date', None),
            'parsed_date': data.get('parsed_date', None),
            'is_incident': data.get('is_incident', None),
            'severity': data.get('severity', None),
            'time_window': data.get('time_window', None),
            'discovery_pipeline': data.get('discovery_pipeline', None),
            'url': data.get('url', None),
            'title': data.get('title', None),
            'description': data.get('description', None),
            'content': data.get('content', None),
            'source': data.get('source', None),
            'validated': data.get('validated', None),
            'downloaded_at': data.get('downloaded_at', None),
            'file_path': str(json_path)
        }
        
        return article
        
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Invalid JSON in {json_path.name}: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] Error reading {json_path.name}: {e}")
        return None


def process_company(company_dir: Path) -> List[Dict[str, Any]]:
    """
    Process all JSON files in a company's ESG category folders.
    Returns list of article records.
    """
    articles = []
    company_name = company_dir.name
    
    for category in ESG_CATEGORIES:
        category_dir = company_dir / category
        
        if not category_dir.exists():
            continue
            
        json_files = list(category_dir.glob("*.json"))
        
        for json_file in json_files:
            article = extract_article_data(json_file)
            if article:
                # Override company name from folder if not in JSON
                if article['company'] is None:
                    article['company'] = company_name.replace('_', ' ')
                # Override category from folder if not in JSON
                if article['category'] is None:
                    article['category'] = category
                articles.append(article)
    
    return articles


def calculate_time_decay(parsed_date: Optional[datetime], 
                         decay_constant: float = 365.0) -> float:
    """
    Calculate exponential time decay weight.
    
    Formula: time_weight = exp(-days_old / decay_constant)
    
    Args:
        parsed_date: The article's date
        decay_constant: Days for weight to decay to ~36.8% (default: 365 days)
    
    Returns:
        Time decay weight between 0 and 1
    """
    if parsed_date is None:
        return 0.0
    
    days_old = (TODAY - parsed_date).days
    
    if days_old < 0:
        days_old = 0  # Future dates treated as today
    
    return np.exp(-days_old / decay_constant)


def parse_severity_score(severity: Any) -> float:
    """
    Parse severity as-is from source data without normalization.

    Returns non-negative float severity; missing/invalid values become 0.
    """
    if severity is None or pd.isna(severity):
        return 0.0
    
    try:
        sev = float(severity)
        return max(sev, 0.0)
    except (ValueError, TypeError):
        return 0.0


def aggregate_articles() -> pd.DataFrame:
    """
    Main aggregation function. Loops through all company directories,
    extracts article data, and creates a flat DataFrame.
    """
    print("=" * 60)
    print("ESG DATA FLATTENING & SCORING")
    print("=" * 60)
    print(f"\nSource Directory: {ARTICLES_DIR}")
    print(f"Today's Date: {TODAY.strftime('%Y-%m-%d')}")
    print("\n" + "-" * 60)
    print("PHASE 1: DATA AGGREGATION")
    print("-" * 60)
    
    all_articles = []
    company_stats = {}
    
    # Get all company directories
    company_dirs = sorted([d for d in ARTICLES_DIR.iterdir() if d.is_dir()])
    total_companies = len(company_dirs)
    
    print(f"\nFound {total_companies} company directories")
    print("\nProcessing companies...")
    
    for idx, company_dir in enumerate(company_dirs, 1):
        company_name = company_dir.name.replace('_', ' ')
        articles = process_company(company_dir)
        
        # Track company stats
        company_stats[company_name] = {
            'folder_name': company_dir.name,
            'total_articles': len(articles),
            'has_data': len(articles) > 0
        }
        
        if articles:
            all_articles.extend(articles)
        
        # Progress update every 50 companies
        if idx % 50 == 0 or idx == total_companies:
            print(f"  Processed {idx}/{total_companies} companies...")
    
    print(f"\nTotal articles extracted: {len(all_articles)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_articles)
    
    return df, company_stats


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering: date parsing, time decay, scoring.
    """
    print("\n" + "-" * 60)
    print("PHASE 2: FEATURE ENGINEERING")
    print("-" * 60)
    
    # Step 1: Parse dates
    print("\n[1/4] Parsing dates...")
    
    # First try parsed_date field, then fall back to date field
    df['parsed_date_final'] = df.apply(
        lambda row: parse_date(row['parsed_date']) if row['parsed_date'] else parse_date(row['date']),
        axis=1
    )
    
    # Calculate days old
    df['days_old'] = df['parsed_date_final'].apply(
        lambda x: (TODAY - x).days if x else None
    )
    
    valid_dates = df['parsed_date_final'].notna().sum()
    print(f"  Valid dates parsed: {valid_dates}/{len(df)} ({100*valid_dates/len(df):.1f}%)")
    
    # Step 2: Calculate time decay
    print("\n[2/4] Calculating time decay weights...")
    df['time_weight'] = df['parsed_date_final'].apply(calculate_time_decay)
    
    # Step 3: Parse severity (raw/original scale)
    print("\n[3/4] Parsing severity scores (no normalization)...")
    df['severity_score'] = df['severity'].apply(parse_severity_score)
    
    # Step 4: NLP Sentiment & Relevance
    print("\n[4/5] Performing NLP Sentiment Analysis...")
    
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df['content'] = df['content'].fillna('')
    df['full_text'] = df['title'] + ". " + df['description'] + ". " + df['content']
    
    def calc_nlp(row):
        text = str(row['full_text']).lower()
        if not text.strip() or text.strip() == ". .":
            return pd.Series({'sentiment_score': 0.0, 'esg_relevance_boost': 1.0})
            
        scores = sia.polarity_scores(text)
        compound = scores['compound']
        
        words = re.findall(r'\b\w+\b', text)
        total_words = len(words)
        boost = 1.0
        if total_words > 0:
            trigger_count = sum(1 for w in words if w in ESG_IMPLICATION_TRIGGERS)
            trigger_density = trigger_count / total_words
            boost = 1.0 + min(0.5, (trigger_density / 0.02) * 0.5)
            
        return pd.Series({'sentiment_score': compound, 'esg_relevance_boost': boost})
        
    nlp_results = df.apply(calc_nlp, axis=1)
    df = pd.concat([df, nlp_results], axis=1)
    
    # Step 5: Calculate event scores
    print("\n[5/5] Calculating event scores...")
    
    # Convert is_incident to numeric (handle None, boolean, string)
    df['is_incident_flag'] = df['is_incident'].apply(
        lambda x: 1 if x in [1, True, '1', 'True', 'true', 'yes', 'Yes'] else 0
    )
    
    # Event score = raw severity * sentiment_multiplier * relevance_boost * time_weight
    def calc_final_score(row):
        if row['is_incident_flag'] != 1:
            return 0.0
        base = row['severity_score']
        time_w = row['time_weight']
        comp = row['sentiment_score']
        boost = row['esg_relevance_boost']
        
        sent_mult = 1.0 - (comp * 0.5)
        
        return base * sent_mult * boost * time_w
        
    df['event_score'] = df.apply(calc_final_score, axis=1)
    
    # Summary stats
    incidents_count = df['is_incident_flag'].sum()
    print(f"\n  Total incidents identified: {incidents_count}")
    print(f"  Average event score (incidents only): {df[df['is_incident_flag']==1]['event_score'].mean():.4f}")
    
    return df


def handle_edge_cases(df: pd.DataFrame, company_stats: Dict) -> pd.DataFrame:
    """
    Handle edge cases for companies:
    
    Case A: Company exists but has no is_incident == True records
            -> Set incident_count = 0, incident_risk_score = 0,
               data_coverage_flag = "no_negative_incidents_found"
    
    Case B: Company has no JSON files at all
            -> Set incident_count = 0, incident_risk_score = 0,
               data_coverage_flag = "no_reported_incidents"
    """
    print("\n" + "-" * 60)
    print("PHASE 3: EDGE CASE HANDLING")
    print("-" * 60)
    
    # Aggregate by company
    print("\n[1/2] Aggregating company-level metrics...")
    
    # Companies in the dataset
    companies_in_df = set(df['company'].unique()) if len(df) > 0 else set()
    
    # Build company summary
    company_summary = []
    
    for company_name, stats in company_stats.items():
        # Get company's articles
        company_df = df[df['company'] == company_name] if len(df) > 0 else pd.DataFrame()
        
        # Count incidents
        if len(company_df) > 0:
            incident_articles = company_df[company_df['is_incident_flag'] == 1]
            incident_count = len(incident_articles)
            incident_risk_score = incident_articles['event_score'].sum()
            total_articles = len(company_df)
            
            # Case A: Has articles but no incidents
            if incident_count == 0:
                data_coverage_flag = "no_negative_incidents_found"
            else:
                data_coverage_flag = "incidents_found"
        else:
            # Case B: No JSON files at all
            incident_count = 0
            incident_risk_score = 0.0
            total_articles = 0
            data_coverage_flag = "no_reported_incidents"
        
        # Category breakdown
        env_count = len(company_df[company_df['category'] == 'Environmental']) if len(company_df) > 0 else 0
        soc_count = len(company_df[company_df['category'] == 'Social']) if len(company_df) > 0 else 0
        gov_count = len(company_df[company_df['category'] == 'Governance']) if len(company_df) > 0 else 0

        # Split data for company/category level summary metrics
        env_df = company_df[company_df['category'] == 'Environmental'] if len(company_df) > 0 else pd.DataFrame()
        soc_df = company_df[company_df['category'] == 'Social'] if len(company_df) > 0 else pd.DataFrame()
        gov_df = company_df[company_df['category'] == 'Governance'] if len(company_df) > 0 else pd.DataFrame()
        
        # Environmental incidents
        env_incidents = company_df[(company_df['category'] == 'Environmental') & (company_df['is_incident_flag'] == 1)] if len(company_df) > 0 else pd.DataFrame()
        env_risk = env_incidents['event_score'].sum() if len(env_incidents) > 0 else 0.0
        
        # Social incidents
        soc_incidents = company_df[(company_df['category'] == 'Social') & (company_df['is_incident_flag'] == 1)] if len(company_df) > 0 else pd.DataFrame()
        soc_risk = soc_incidents['event_score'].sum() if len(soc_incidents) > 0 else 0.0
        
        # Governance incidents
        gov_incidents = company_df[(company_df['category'] == 'Governance') & (company_df['is_incident_flag'] == 1)] if len(company_df) > 0 else pd.DataFrame()
        gov_risk = gov_incidents['event_score'].sum() if len(gov_incidents) > 0 else 0.0
        
        # Calculate weighted ESG risk score
        env_weighted_score = env_risk * ESG_WEIGHTS["Environmental"]
        soc_weighted_score = soc_risk * ESG_WEIGHTS["Social"]
        gov_weighted_score = gov_risk * ESG_WEIGHTS["Governance"]
        weighted_esg_risk = (
            env_weighted_score +
            soc_weighted_score +
            gov_weighted_score
        )

        # Age (days old), time weight, and severity summaries
        age_avg = float(company_df['days_old'].mean()) if len(company_df) > 0 and company_df['days_old'].notna().any() else 0.0
        age_env_avg = float(env_df['days_old'].mean()) if len(env_df) > 0 and env_df['days_old'].notna().any() else 0.0
        age_soc_avg = float(soc_df['days_old'].mean()) if len(soc_df) > 0 and soc_df['days_old'].notna().any() else 0.0
        age_gov_avg = float(gov_df['days_old'].mean()) if len(gov_df) > 0 and gov_df['days_old'].notna().any() else 0.0

        time_weight_avg = float(company_df['time_weight'].mean()) if len(company_df) > 0 and company_df['time_weight'].notna().any() else 0.0
        time_weight_env_avg = float(env_df['time_weight'].mean()) if len(env_df) > 0 and env_df['time_weight'].notna().any() else 0.0
        time_weight_soc_avg = float(soc_df['time_weight'].mean()) if len(soc_df) > 0 and soc_df['time_weight'].notna().any() else 0.0
        time_weight_gov_avg = float(gov_df['time_weight'].mean()) if len(gov_df) > 0 and gov_df['time_weight'].notna().any() else 0.0

        severity_avg = float(company_df['severity_score'].mean()) if len(company_df) > 0 and company_df['severity_score'].notna().any() else 0.0
        severity_env_avg = float(env_df['severity_score'].mean()) if len(env_df) > 0 and env_df['severity_score'].notna().any() else 0.0
        severity_soc_avg = float(soc_df['severity_score'].mean()) if len(soc_df) > 0 and soc_df['severity_score'].notna().any() else 0.0
        severity_gov_avg = float(gov_df['severity_score'].mean()) if len(gov_df) > 0 and gov_df['severity_score'].notna().any() else 0.0
        
        company_summary.append({
            'company': company_name,
            'folder_name': stats['folder_name'],
            'total_articles': total_articles,
            'incident_count': incident_count,
            'raw_incident_risk_score': round(incident_risk_score, 4),
            'environmental_articles': env_count,
            'environmental_risk': round(env_risk, 4),
            'social_articles': soc_count,
            'social_risk': round(soc_risk, 4),
            'governance_articles': gov_count,
            'governance_risk': round(gov_risk, 4),
            'final_weighted_esg_score': round(weighted_esg_risk, 4),
            'article_age_days_avg': round(age_avg, 2),
            'article_age_days_environmental_avg': round(age_env_avg, 2),
            'article_age_days_social_avg': round(age_soc_avg, 2),
            'article_age_days_governance_avg': round(age_gov_avg, 2),
            'time_weight_avg': round(time_weight_avg, 4),
            'time_weight_environmental_avg': round(time_weight_env_avg, 4),
            'time_weight_social_avg': round(time_weight_soc_avg, 4),
            'time_weight_governance_avg': round(time_weight_gov_avg, 4),
            'severity_avg': round(severity_avg, 4),
            'severity_environmental_avg': round(severity_env_avg, 4),
            'severity_social_avg': round(severity_soc_avg, 4),
            'severity_governance_avg': round(severity_gov_avg, 4),
            'environmental_weight': ESG_WEIGHTS["Environmental"],
            'social_weight': ESG_WEIGHTS["Social"],
            'governance_weight': ESG_WEIGHTS["Governance"],
            'environmental_raw_score': round(env_risk, 4),
            'social_raw_score': round(soc_risk, 4),
            'governance_raw_score': round(gov_risk, 4),
            'raw_score': round(incident_risk_score, 4),
            'environmental_weighted_score': round(env_weighted_score, 4),
            'social_weighted_score': round(soc_weighted_score, 4),
            'governance_weighted_score': round(gov_weighted_score, 4),
            'final_walk_esg_risk': round(weighted_esg_risk, 4),
            'data_coverage_flag': data_coverage_flag
        })
    
    company_summary_df = pd.DataFrame(company_summary)

    # Walk analysis score is a penalty/risk score from negative incidents only.
    # Keep original ESG category weights and do not transform to rating bands.
    company_summary_df['final_walk_esg_score'] = company_summary_df['final_walk_esg_risk'].round(4)
    
    # Edge case statistics
    print("\n[2/2] Edge case statistics:")
    case_a = len(company_summary_df[company_summary_df['data_coverage_flag'] == 'no_negative_incidents_found'])
    case_b = len(company_summary_df[company_summary_df['data_coverage_flag'] == 'no_reported_incidents'])
    incidents_found = len(company_summary_df[company_summary_df['data_coverage_flag'] == 'incidents_found'])
    
    print(f"  Case A (No negative incidents found): {case_a} companies")
    print(f"  Case B (No reported incidents/data): {case_b} companies")
    print(f"  Companies with incidents: {incidents_found}")
    
    return company_summary_df


def build_walk_data(company_summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ordered walk-style ESG score table requested for final Excel output.
    """
    walk_columns_in_order = [
        'company',
        'total_articles',
        'environmental_articles',
        'social_articles',
        'governance_articles',
        'article_age_days_avg',
        'article_age_days_environmental_avg',
        'article_age_days_social_avg',
        'article_age_days_governance_avg',
        'time_weight_avg',
        'time_weight_environmental_avg',
        'time_weight_social_avg',
        'time_weight_governance_avg',
        'severity_avg',
        'severity_environmental_avg',
        'severity_social_avg',
        'severity_governance_avg',
        'environmental_weight',
        'social_weight',
        'governance_weight',
        'environmental_raw_score',
        'social_raw_score',
        'governance_raw_score',
        'raw_score',
        'environmental_weighted_score',
        'social_weighted_score',
        'governance_weighted_score',
        'final_walk_esg_score',
        'incident_count',
        'data_coverage_flag'
    ]

    available_columns = [c for c in walk_columns_in_order if c in company_summary_df.columns]
    walk_df = company_summary_df[available_columns].copy()

    return walk_df


def save_outputs(df: pd.DataFrame, company_summary_df: pd.DataFrame):
    """
    Save the final outputs to CSV files.
    """
    print("\n" + "-" * 60)
    print("PHASE 4: SAVING OUTPUTS")
    print("-" * 60)
    
    # Prepare final article DataFrame
    output_columns = [
        'company', 'category', 'title', 'url', 'date', 'parsed_date_final',
        'days_old', 'time_weight', 'is_incident_flag', 'severity', 
        'severity_score', 'sentiment_score', 'esg_relevance_boost', 
        'event_score', 'time_window', 'discovery_pipeline', 'source', 
        'validated', 'downloaded_at'
    ]
    
    # Only select columns that exist
    available_columns = [c for c in output_columns if c in df.columns]
    df_output = df[available_columns].copy()
    
    # Rename parsed_date_final to parsed_date for cleaner output
    if 'parsed_date_final' in df_output.columns:
        df_output.rename(columns={'parsed_date_final': 'parsed_date'}, inplace=True)
    
    # Save article-level data
    print(f"\n[1/3] Saving article-level data to: {OUTPUT_FILE}")
    df_output.to_csv(OUTPUT_FILE, index=False)
    print(f"  Records saved: {len(df_output)}")
    
    # Save company summary
    print(f"\n[2/3] Saving company summary to: {COMPANY_SUMMARY_FILE}")
    company_summary_df.to_csv(COMPANY_SUMMARY_FILE, index=False)
    print(f"  Companies saved: {len(company_summary_df)}")

    # Save single walk-data Excel requested for ESG score breakdown
    walk_df = build_walk_data(company_summary_df)
    print(f"\n[3/3] Saving walk ESG data to Excel: {WALK_DATA_EXCEL_FILE}")
    with pd.ExcelWriter(WALK_DATA_EXCEL_FILE, engine='openpyxl') as writer:
        walk_df.to_excel(writer, index=False, sheet_name='walk_data_esg_score')
    print(f"  Companies saved: {len(walk_df)}")


def print_summary_statistics(df: pd.DataFrame, company_summary_df: pd.DataFrame):
    """
    Print final summary statistics.
    """
    print("\n" + "=" * 60)
    print("FINAL SUMMARY STATISTICS")
    print("=" * 60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Total companies: {len(company_summary_df)}")
    print(f"  Total articles: {len(df)}")
    print(f"  Total incidents: {df['is_incident_flag'].sum()}")
    
    print(f"\n📈 Category Breakdown:")
    if len(df) > 0:
        for cat in ESG_CATEGORIES:
            cat_count = len(df[df['category'] == cat])
            cat_pct = 100 * cat_count / len(df)
            print(f"  {cat}: {cat_count} articles ({cat_pct:.1f}%)")
    
    print(f"\n⚠️ Risk Distribution:")
    print(f"  Companies with incidents: {len(company_summary_df[company_summary_df['incident_count'] > 0])}")
    print(f"  Companies without incidents: {len(company_summary_df[company_summary_df['incident_count'] == 0])}")
    
    if len(company_summary_df[company_summary_df['raw_incident_risk_score'] > 0]) > 0:
        top_risk = company_summary_df.nlargest(5, 'raw_incident_risk_score')[['company', 'incident_count', 'raw_incident_risk_score', 'final_weighted_esg_score']]
        print(f"\n🔝 Top 5 Companies by Raw Risk Score:")
        for _, row in top_risk.iterrows():
            print(f"  {row['company']}: {row['incident_count']} incidents, raw_risk={row['raw_incident_risk_score']:.4f}, final_weighted_esg={row['final_weighted_esg_score']:.4f}")
    
    print(f"\n📁 Output Files:")
    print(f"  1. {OUTPUT_FILE}")
    print(f"  2. {COMPANY_SUMMARY_FILE}")
    print(f"  3. {WALK_DATA_EXCEL_FILE}")


def main():
    """
    Main execution function.
    """
    try:
        # Phase 1: Aggregate data
        df, company_stats = aggregate_articles()
        
        if len(df) == 0:
            print("\n[WARNING] No articles found! Creating empty outputs with all companies.")
            df = pd.DataFrame()
        
        # Phase 2: Feature engineering
        if len(df) > 0:
            df = engineer_features(df)
        else:
            # Add required columns for empty DataFrame
            df = pd.DataFrame(columns=[
                'company', 'category', 'title', 'url', 'date', 'parsed_date_final',
                'days_old', 'time_weight', 'is_incident_flag', 'severity', 
                'severity_score', 'sentiment_score', 'esg_relevance_boost',
                'event_score', 'time_window', 'discovery_pipeline', 'source', 
                'validated', 'downloaded_at'
            ])
        
        # Phase 3: Handle edge cases
        company_summary_df = handle_edge_cases(df, company_stats)
        
        # Phase 4: Save outputs
        save_outputs(df, company_summary_df)
        
        # Print summary
        print_summary_statistics(df, company_summary_df)
        
        print("\n" + "=" * 60)
        print("✅ ESG DATA FLATTENING COMPLETE!")
        print("=" * 60)
        
        return df, company_summary_df
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    df, company_summary = main()
