"""
ESG Article Downloader v3 - Strict Incident + Broad Discovery
===============================================================
Downloads relevant ESG news articles for Indian companies.

This system combines:
- A high-precision ESG incident extractor (strict pipeline)
- A high-recall ESG discovery layer using modern ESG terminology
- A 5-year rolling window for temporal relevance

Key Features:
- Strict company name matching in article title/content
- ESG keyword validation to filter out financial news
- Severity-based prioritization for incidents
- Broad ESG discovery for sustainability, governance, social initiatives
- 5-year rolling window filter
- Better search queries with India-specific sources

This design mitigates ESG sparsity bias while preserving signal quality
for credit risk modeling.

Max 7 articles per category (E, S, G) = 21 per company
Less is fine if fewer relevant articles exist
"""

import os
import json
import time
import re
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote_plus
import requests
from bs4 import BeautifulSoup
import random
import warnings

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Try to import dateutil for better date parsing
try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    print("Warning: python-dateutil not installed. Using basic date parsing.")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = r"c:\Users\Lenovo\Documents\Python\Capstone"
ARTICLES_DIR = os.path.join(BASE_DIR, "articles")
PROGRESS_FILE = os.path.join(BASE_DIR, "download_progress_v3.json")

# Max 7 per category, but only save if truly relevant
MAX_PER_CATEGORY = 7

# 5-year rolling window (in days) - primary
TIME_WINDOW_YEARS = 5
TIME_WINDOW_DAYS = TIME_WINDOW_YEARS * 365

# 10-year extended window (fallback if no articles found)
EXTENDED_WINDOW_YEARS = 10
EXTENDED_WINDOW_DAYS = EXTENDED_WINDOW_YEARS * 365

# Rate limiting
REQUEST_DELAY = 2
BATCH_DELAY = 3

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edge/119.0.0.0',
]

# ============================================================================
# COMPANY LIST WITH SEARCH ALIASES
# ============================================================================

COMPANY_ALIASES = {
    "360 ONE WAM LTD": ["360 ONE", "360ONE", "IIFL Wealth"],
    "3M INDIA LTD": ["3M India"],
    "AADHAR HOUSING FINANCE LTD": ["Aadhar Housing", "Aadhar Finance"],
    "AARTI INDUSTRIES LIMITED": ["Aarti Industries"],
    "AAVAS FINANCIERS LTD": ["Aavas Financiers"],
    "ABB INDIA LTD": ["ABB India"],
    "ABBOTT INDIA LTD": ["Abbott India"],
    "ACC LTD": ["ACC Cement", "ACC Limited"],
    "ACME SOLAR HOLDINGS LTD": ["ACME Solar"],
    "ACTION CONSTRUCTION EQUIPMEN": ["Action Construction", "ACE"],
    "ADANI ENERGY SOLUTIONS LTD": ["Adani Energy", "Adani Transmission"],
    "ADANI ENTERPRISES LTD": ["Adani Enterprises", "Adani Group"],
    "ADANI GREEN ENERGY LTD": ["Adani Green", "Adani Green Energy"],
    "ADANI PORTS AND SPECIAL ECON": ["Adani Ports", "APSEZ"],
    "ADANI POWER LTD": ["Adani Power"],
    "ADANI TOTAL GAS LTD": ["Adani Total Gas", "Adani Gas"],
    "ADITYA BIRLA CAPITAL LTD": ["Aditya Birla Capital", "AB Capital"],
    "ADITYA BIRLA FASHION AND RET": ["Aditya Birla Fashion", "ABFRL"],
    "AMBUJA CEMENTS LTD": ["Ambuja Cements", "Ambuja Cement"],
    "APOLLO HOSPITALS ENTERPRISE": ["Apollo Hospitals", "Apollo Hospital"],
    "ASIAN PAINTS LTD": ["Asian Paints"],
    "AXIS BANK LTD": ["Axis Bank"],
    "BAJAJ AUTO LTD": ["Bajaj Auto"],
    "BAJAJ FINANCE LTD": ["Bajaj Finance"],
    "BAJAJ FINSERV LTD": ["Bajaj Finserv"],
    "BANK OF BARODA": ["Bank of Baroda", "BoB"],
    "BANK OF INDIA": ["Bank of India", "BOI"],
    "BHARTI AIRTEL LTD": ["Bharti Airtel", "Airtel"],
    "BHARAT PETROLEUM CORP LTD": ["BPCL", "Bharat Petroleum"],
    "BIOCON LTD": ["Biocon"],
    "BOSCH LTD": ["Bosch India"],
    "BRITANNIA INDUSTRIES LTD": ["Britannia Industries", "Britannia"],
    "CANARA BANK": ["Canara Bank"],
    "CIPLA LTD": ["Cipla"],
    "COAL INDIA LTD": ["Coal India", "CIL"],
    "COLGATE PALMOLIVE (INDIA)": ["Colgate India", "Colgate Palmolive India"],
    "DABUR INDIA LTD": ["Dabur India", "Dabur"],
    "DIVI'S LABORATORIES LTD": ["Divis Labs", "Divi's Laboratories"],
    "DLF LTD": ["DLF"],
    "DR. REDDY'S LABORATORIES": ["Dr Reddy's", "Dr Reddys Labs"],
    "EICHER MOTORS LTD": ["Eicher Motors", "Royal Enfield"],
    "GAIL INDIA LTD": ["GAIL India", "GAIL"],
    "GODREJ CONSUMER PRODUCTS LTD": ["Godrej Consumer", "GCPL"],
    "GODREJ PROPERTIES LTD": ["Godrej Properties"],
    "GRASIM INDUSTRIES LTD": ["Grasim Industries", "Grasim"],
    "HCL TECHNOLOGIES LTD": ["HCL Tech", "HCL Technologies"],
    "HDFC BANK LIMITED": ["HDFC Bank"],
    "HDFC LIFE INSURANCE CO LTD": ["HDFC Life"],
    "HERO MOTOCORP LTD": ["Hero MotoCorp", "Hero Honda"],
    "HINDALCO INDUSTRIES LTD": ["Hindalco", "Hindalco Industries"],
    "HINDUSTAN AERONAUTICS LTD": ["HAL", "Hindustan Aeronautics"],
    "HINDUSTAN PETROLEUM CORP": ["HPCL", "Hindustan Petroleum"],
    "HINDUSTAN UNILEVER LTD": ["HUL", "Hindustan Unilever"],
    "HINDUSTAN ZINC LTD": ["Hindustan Zinc"],
    "ICICI BANK LTD": ["ICICI Bank"],
    "ICICI LOMBARD GENERAL INSURA": ["ICICI Lombard"],
    "ICICI PRUDENTIAL LIFE INSURA": ["ICICI Prudential"],
    "IDBI BANK LTD": ["IDBI Bank"],
    "IDFC FIRST BANK LTD": ["IDFC First Bank"],
    "INDIAN HOTELS CO LTD": ["Indian Hotels", "Taj Hotels", "IHCL"],
    "INDIAN OIL CORP LTD": ["Indian Oil", "IOCL", "IOC"],
    "INDUSIND BANK LTD": ["IndusInd Bank"],
    "INFOSYS LTD": ["Infosys"],
    "INTERGLOBE AVIATION LTD": ["InterGlobe Aviation", "IndiGo"],
    "ITC LTD": ["ITC Limited", "ITC"],
    "JINDAL STEEL LTD": ["Jindal Steel", "JSPL"],
    "JSW STEEL LTD": ["JSW Steel"],
    "JSW ENERGY LTD": ["JSW Energy"],
    "KOTAK MAHINDRA BANK LTD": ["Kotak Bank", "Kotak Mahindra"],
    "LARSEN & TOUBRO LTD": ["L&T", "Larsen Toubro"],
    "LIC HOUSING FINANCE LTD": ["LIC Housing"],
    "LIFE INSURANCE CORPORATION": ["LIC", "Life Insurance Corporation"],
    "LUPIN LTD": ["Lupin"],
    "MAHINDRA & MAHINDRA LTD": ["Mahindra", "M&M", "Mahindra and Mahindra"],
    "MARICO LTD": ["Marico"],
    "MARUTI SUZUKI INDIA LTD": ["Maruti Suzuki", "Maruti"],
    "NESTLE INDIA LTD": ["Nestle India"],
    "NMDC LTD": ["NMDC"],
    "NTPC LTD": ["NTPC"],
    "OIL & NATURAL GAS CORP LTD": ["ONGC", "Oil and Natural Gas"],
    "OIL INDIA LTD": ["Oil India"],
    "ONE 97 COMMUNICATIONS LTD": ["Paytm", "One97"],
    "PIDILITE INDUSTRIES LTD": ["Pidilite", "Fevicol"],
    "POWER GRID CORP OF INDIA LTD": ["Power Grid", "PGCIL"],
    "PUNJAB NATIONAL BANK": ["PNB", "Punjab National Bank"],
    "RELIANCE INDUSTRIES LIMITED": ["Reliance Industries", "RIL", "Reliance"],
    "SBI LIFE INSURANCE CO LTD": ["SBI Life"],
    "SHREE CEMENT LTD": ["Shree Cement"],
    "SIEMENS LTD": ["Siemens India"],
    "STATE BANK OF INDIA": ["SBI", "State Bank"],
    "STEEL AUTHORITY OF INDIA": ["SAIL", "Steel Authority"],
    "SUN PHARMACEUTICAL INDUS": ["Sun Pharma", "Sun Pharmaceutical"],
    "TATA CHEMICALS LTD": ["Tata Chemicals"],
    "TATA COMMUNICATIONS LTD": ["Tata Communications"],
    "TATA CONSULTANCY SVCS LTD": ["TCS", "Tata Consultancy"],
    "TATA CONSUMER PRODUCTS LTD": ["Tata Consumer", "Tata Tea"],
    "TATA MOTORS PASSENGER VEHICL": ["Tata Motors"],
    "TATA POWER CO LTD": ["Tata Power"],
    "TATA STEEL LTD": ["Tata Steel"],
    "TECH MAHINDRA LTD": ["Tech Mahindra"],
    "TITAN CO LTD": ["Titan", "Titan Company"],
    "TORRENT PHARMACEUTICALS LTD": ["Torrent Pharma"],
    "TORRENT POWER LTD": ["Torrent Power"],
    "ULTRATECH CEMENT LTD": ["UltraTech Cement", "Ultratech"],
    "UNITED SPIRITS LTD": ["United Spirits", "Diageo India"],
    "UPL LTD": ["UPL"],
    "VEDANTA LTD": ["Vedanta"],
    "VODAFONE IDEA LTD": ["Vodafone Idea", "Vi", "Voda Idea"],
    "WIPRO LTD": ["Wipro"],
    "YES BANK LTD": ["Yes Bank"],
    "ZEE ENTERTAINMENT ENTERPRISE": ["Zee Entertainment", "Zee TV", "ZEEL"],
    "ZYDUS LIFESCIENCES LTD": ["Zydus Lifesciences", "Zydus Cadila", "Cadila"],
}

# Full company list (500 companies)
COMPANIES = [
    "360 ONE WAM LTD", "3M INDIA LTD", "AADHAR HOUSING FINANCE LTD", "AARTI INDUSTRIES LIMITED",
    "AAVAS FINANCIERS LTD", "ABB INDIA LTD", "ABBOTT INDIA LTD", "ACC LTD",
    "ACME SOLAR HOLDINGS LTD", "ACTION CONSTRUCTION EQUIPMEN", "ADANI ENERGY SOLUTIONS LTD",
    "ADANI ENTERPRISES LTD", "ADANI GREEN ENERGY LTD", "ADANI PORTS AND SPECIAL ECON",
    "ADANI POWER LTD", "ADANI TOTAL GAS LTD", "ADITYA BIRLA CAPITAL LTD",
    "ADITYA BIRLA FASHION AND RET", "ADITYA BIRLA LIFESTYLE BRAND", "ADITYA BIRLA REAL ESTATE LTD",
    "ADITYA BIRLA SUN LIFE AMC LT", "AEGIS LOGISTICS LTD", "AEGIS VOPAK TERMINALS LTD",
    "AFCONS INFRASTUCTURE LTD", "AFFLE 3I LTD", "AIA ENGINEERING LTD", "AJANTA PHARMA LTD",
    "AKUMS DRUGS & PHARMACEUTICAL", "AKZO NOBEL INDIA LTD", "ALEMBIC PHARMACEUTICALS LTD",
    "ALKEM LABORATORIES LTD", "ALKYL AMINES CHEMICALS LTD", "ALOK INDUSTRIES LTD",
    "AMARA RAJA ENERGY & MOBILITY", "AMBER ENTERPRISES INDIA LTD", "AMBUJA CEMENTS LTD",
    "ANAND RATHI WEALTH LTD", "ANANT RAJ LTD", "ANGEL ONE LTD", "APAR INDUSTRIES LTD",
    "APL APOLLO TUBES LTD", "APOLLO HOSPITALS ENTERPRISE", "APOLLO TYRES LTD",
    "APTUS VALUE HOUSING FINANCE", "ASAHI INDIA GLASS LTD", "ASHOK LEYLAND LTD",
    "ASIAN PAINTS LTD", "ASTER DM HEALTHCARE LTD", "ASTRAL LTD", "ASTRAZENECA PHARMA INDIA LTD",
    "ATHER ENERGY LTD", "ATUL LTD", "AU SMALL FINANCE BANK LTD", "AUROBINDO PHARMA LTD",
    "AUTHUM INVESTMENT INFRASTUCT", "AVENUE SUPERMARTS LTD", "AWL AGRI BUSINESS LTD",
    "AXIS BANK LTD", "BAJAJ AUTO LTD", "BAJAJ FINANCE LTD", "BAJAJ FINSERV LTD",
    "BAJAJ HOLDINGS AND INVESTMEN", "BAJAJ HOUSING FINANCE LTD", "BALKRISHNA INDUSTRIES LTD",
    "BALRAMPUR CHINI MILLS LTD", "BANDHAN BANK LTD", "BANK OF BARODA", "BANK OF INDIA",
    "BANK OF MAHARASHTRA", "BASF INDIA LTD", "BATA INDIA LTD", "BAYER CROPSCIENCE LTD",
    "BEML LTD", "BERGER PAINTS INDIA LTD", "BHARAT DYNAMICS LTD", "BHARAT ELECTRONICS LTD",
    "BHARAT FORGE LTD", "BHARAT HEAVY ELECTRICALS", "BHARAT PETROLEUM CORP LTD",
    "BHARTI AIRTEL LTD", "BHARTI HEXACOM LTD", "BIKAJI FOODS INTERNATIONAL L", "BIOCON LTD",
    "BIRLASOFT LTD", "BLS INTERNATIONAL LTD", "BLUE DART EXPRESS LTD", "BLUE JET HEALTHCARE LTD",
    "BLUE STAR LTD", "BOMBAY BURMAH TRADING CORP", "BOSCH LTD", "BRAINBEES SOLUTIONS LTD",
    "BRIGADE ENTERPRISES LTD", "BRITANNIA INDUSTRIES LTD", "BSE LTD", "CAMPUS ACTIVEWEAR LTD",
    "CAN FIN HOMES LTD", "CANARA BANK", "CAPLIN POINT LABORATORIES", "CAPRI GLOBAL CAPITAL LTD",
    "CARBORUNDUM UNIVERSAL LTD", "CASTROL INDIA LTD", "CCL PRODUCTS INDIA LTD",
    "CE INFO SYSTEMS LTD", "CEAT LTD", "CENTRAL BANK OF INDIA", "CENTRAL DEPOSITORY SERVICES",
    "CENTURY PLYBOARDS INDIA LTD", "CERA SANITARYWARE LTD", "CESC LTD",
    "CG POWER AND INDUSTRIAL SOLU", "CHALET HOTELS LTD", "CHAMBAL FERTILISERS & CHEM",
    "CHENNAI PETROLEUM CORP LTD", "CHOICE INTERNATIONAL LTD", "CHOLAMANDALAM FINANCIAL HOLD",
    "CHOLAMANDALAM INVESTMENT AND", "CIPLA LTD", "CITY UNION BANK LTD",
    "CLEAN SCIENCE & TECHNOLOGY L", "COAL INDIA LTD", "COCHIN SHIPYARD LTD", "COFORGE LIMITED",
    "COHANCE LIFESCIENCES LTD", "COLGATE PALMOLIVE (INDIA)", "COMPUTER AGE MANAGEMENT SERV",
    "CONCORD BIOTECH LTD", "CONTAINER CORP OF INDIA LTD", "COROMANDEL INTERNATIONAL LTD",
    "CRAFTSMAN AUTOMATION LTD", "CREDITACCESS GRAMEEN LTD", "CRISIL LTD",
    "CROMPTON GREAVES CONSUMER EL", "CUMMINS INDIA LTD", "CYIENT LTD", "DABUR INDIA LTD",
    "DALMIA BHARAT LTD", "DATA PATTERNS INDIA LTD", "DCM SHRIRAM LTD",
    "DEEPAK FERTILISERS & PETRO", "DEEPAK NITRITE LTD", "DELHIVERY LTD",
    "DEVYANI INTERNATIONAL LTD", "DIVI'S LABORATORIES LTD", "DIXON TECHNOLOGIES INDIA LTD",
    "DLF LTD", "DOMS INDUSTRIES LTD", "DR AGARWAL'S HEALTH CARE LTD", "DR LAL PATHLABS LTD",
    "DR. REDDY'S LABORATORIES", "ECLERX SERVICES LTD", "EICHER MOTORS LTD",
    "EID PARRY INDIA LTD", "EIH LTD", "ELECON ENGINEERING CO LTD", "ELGI EQUIPMENTS LTD",
    "EMAMI LTD", "EMCURE PHARMACEUTICALS LTD", "ENDURANCE TECHNOLOGIES LTD",
    "ENGINEERS INDIA LTD", "ERIS LIFESCIENCES LTD", "ESCORTS KUBOTA LTD", "ETERNAL LTD",
    "EXIDE INDUSTRIES LTD", "FEDERAL BANK LTD", "FERTILISERS & CHEM TRAVANCR",
    "FINOLEX CABLES LTD", "FINOLEX INDUSTRIES LTD", "FIRSTSOURCE SOLUTIONS LTD",
    "FIVE-STAR BUSINESS FINANCE L", "FORCE MOTORS LTD", "FORTIS HEALTHCARE LTD",
    "FSN E-COMMERCE VENTURES LTD", "GAIL INDIA LTD", "GARDEN REACH SHIPBUILDERS &",
    "GE VERNOVA T&D INDIA LTD", "GENERAL INS CORP OF INDIA", "GILLETTE INDIA LTD",
    "GLAND PHARMA LTD", "GLAXOSMITHKLINE PHARMACEUTIC", "GLENMARK PHARMACEUTICALS LTD",
    "GLOBAL HEALTH LTD/INDIA", "GMR AIRPORTS LTD", "GO DIGIT GENERAL INSURANCE L",
    "GODAWARI POWER AND ISPAT LTD", "GODFREY PHILLIPS INDIA LTD", "GODREJ AGROVET LTD",
    "GODREJ CONSUMER PRODUCTS LTD", "GODREJ INDUSTRIES LTD", "GODREJ PROPERTIES LTD",
    "GRANULES INDIA LTD", "GRAPHITE INDIA LTD", "GRASIM INDUSTRIES LTD", "GRAVITA INDIA LTD",
    "GREAT EASTERN SHIPPING CO", "GUJARAT FLUOROCHEMICALS LTD", "GUJARAT GAS LTD",
    "GUJARAT MINERAL DEV CORP LTD", "GUJARAT STATE PETRONET LTD", "HAPPIEST MINDS TECHNOLOGIES",
    "HAVELLS INDIA LTD", "HBL ENGINEERING LTD", "HCL TECHNOLOGIES LTD",
    "HDFC ASSET MANAGEMENT CO LTD", "HDFC BANK LIMITED", "HDFC LIFE INSURANCE CO LTD",
    "HEG LTD", "HERO MOTOCORP LTD", "HEXAWARE TECHNOLOGIES LTD", "HFCL LTD",
    "HIMADRI SPECIALITY CHEMICAL", "HINDALCO INDUSTRIES LTD", "HINDUSTAN AERONAUTICS LTD",
    "HINDUSTAN COPPER LTD", "HINDUSTAN PETROLEUM CORP", "HINDUSTAN UNILEVER LTD",
    "HINDUSTAN ZINC LTD", "HITACHI ENERGY INDIA LTD", "HOME FIRST FINANCE CO INDIA",
    "HONASA CONSUMER LTD", "HONEYWELL AUTOMATION INDIA", "HOUSING & URBAN DEV CORP LTD",
    "HYUNDAI MOTOR INDIA LTD", "ICICI BANK LTD", "ICICI LOMBARD GENERAL INSURA",
    "ICICI PRUDENTIAL LIFE INSURA", "IDBI BANK LTD", "IDFC FIRST BANK LTD", "IFCI LTD",
    "IIFL FINANCE LTD", "INDEGENE LTD", "INDIA CEMENTS LTD", "INDIAMART INTERMESH LTD",
    "INDIAN BANK", "INDIAN ENERGY EXCHANGE LTD", "INDIAN HOTELS CO LTD", "INDIAN OIL CORP LTD",
    "INDIAN OVERSEAS BANK", "INDIAN RAILWAY CATERING & TO", "INDIAN RAILWAY FINANCE CORP",
    "INDIAN RENEWABLE ENERGY DEVE", "INDRAPRASTHA GAS LTD", "INDUS TOWERS LTD",
    "INDUSIND BANK LTD", "INFO EDGE INDIA LTD", "INFOSYS LTD", "INOX INDIA LTD",
    "INOX WIND LTD", "INTELLECT DESIGN ARENA LTD", "INTERGLOBE AVIATION LTD",
    "INTERNATIONAL GEMMOLOGICAL I", "INVENTURUS KNOWLEDGE SOLUTIO", "IPCA LABORATORIES LTD",
    "IRB INFRASTRUCTURE DEVELOPER", "IRCON INTERNATIONAL LTD", "ITC HOTELS LIMITED", "ITC LTD",
    "ITI LTD", "J.B. CHEMICALS & PHARMA LTD", "JAIPRAKASH POWER VENTURES LT",
    "JAMMU & KASHMIR BANK LTD", "JBM AUTO LTD", "JINDAL SAW LTD", "JINDAL STAINLESS LTD",
    "JINDAL STEEL LTD", "JIO FINANCIAL SERVICES LTD", "JK CEMENT LTD", "JK TYRE & INDUSTRIES LTD",
    "JM FINANCIAL LTD", "JSW CEMENT LTD", "JSW ENERGY LTD", "JSW INFRASTRUCTURE LTD",
    "JSW STEEL LTD", "JUBILANT FOODWORKS LTD", "JUBILANT INGREVIA LTD", "JUBILANT PHARMOVA LTD",
    "JUPITER WAGONS LTD", "JYOTHY LABS LTD", "JYOTI CNC AUTOMATION LTD", "KAJARIA CERAMICS LTD",
    "KALPATARU PROJECTS INTERNATI", "KALYAN JEWELLERS INDIA LTD", "KARUR VYSYA BANK LTD",
    "KAYNES TECHNOLOGY INDIA LTD", "KEC INTERNATIONAL LTD", "KEI INDUSTRIES LTD",
    "KFIN TECHNOLOGIES LTD", "KIRLOSKAR BROTHERS LTD", "KIRLOSKAR OIL ENGINES LTD",
    "KOTAK MAHINDRA BANK LTD", "KPIT TECHNOLOGIES LTD", "KPR MILL LTD",
    "KRISHNA INSTITUTE OF MEDICAL", "KSB LTD", "L&T FINANCE LTD", "L&T TECHNOLOGY SERVICES LTD",
    "LARSEN & TOUBRO LTD", "LATENT VIEW ANALYTICS LTD", "LAURUS LABS LTD",
    "LEELA PALACES HOTELS & RESOR", "LEMON TREE HOTELS LTD", "LIC HOUSING FINANCE LTD",
    "LIFE INSURANCE CORPORATION", "LINDE INDIA LTD", "LLOYDS METALS & ENERGY LTD",
    "LODHA DEVELOPERS LTD", "LT FOODS LTD", "LTIMINDTREE LTD", "LUPIN LTD", "MAHANAGAR GAS LTD",
    "MAHARASHTRA SCOOTERS LTD", "MAHARASHTRA SEAMLESS LTD", "MAHINDRA & MAHINDRA FIN SECS",
    "MAHINDRA & MAHINDRA LTD", "MANAPPURAM FINANCE LTD", "MANGALORE REFINERY & PETRO",
    "MANKIND PHARMA LTD", "MARICO LTD", "MARUTI SUZUKI INDIA LTD", "MAX FINANCIAL SERVICES LTD",
    "MAX HEALTHCARE INSTITUTE LTD", "MAZAGON DOCK SHIPBUILDERS LT", "METROPOLIS HEALTHCARE LTD",
    "MINDA CORP LTD", "MMTC LTD", "MOTHERSON SUMI WIRING INDIA", "MOTILAL OSWAL FINANCIAL SERV",
    "MPHASIS LTD", "MRF LTD", "MULTI COMMODITY EXCH INDIA", "MUTHOOT FINANCE LTD",
    "NARAYANA HRUDAYALAYA LTD", "NATCO PHARMA LTD", "NATIONAL ALUMINIUM CO LTD", "NAVA LTD",
    "NAVIN FLUORINE INTERNATIONAL", "NBCC INDIA LTD", "NCC LTD", "NESTLE INDIA LTD",
    "NETWEB TECHNOLOGIES INDIA LT", "NEULAND LABORATORIES LTD", "NEW INDIA ASSURANCE CO LTD/T",
    "NEWGEN SOFTWARE TECHNOLOGIES", "NHPC LTD", "NIPPON LIFE INDIA ASSET MANA",
    "NIVA BUPA HEALTH INSURANCE C", "NLC INDIA LTD", "NMDC LTD", "NMDC STEEL LTD",
    "NTPC GREEN ENERGY LTD", "NTPC LTD", "NUVAMA WEALTH MANAGEMENT LTD", "NUVOCO VISTAS CORP LTD",
    "OBEROI REALTY LTD", "OIL & NATURAL GAS CORP LTD", "OIL INDIA LTD",
    "OLA ELECTRIC MOBILITY LTD", "OLECTRA GREENTECH LTD", "ONE 97 COMMUNICATIONS LTD",
    "ONESOURCE SPECIALTY PHARMA L", "ORACLE FINANCIAL SERVICES", "PAGE INDUSTRIES LTD",
    "PATANJALI FOODS LTD", "PB FINTECH LTD", "PCBL CHEMICAL LTD", "PERSISTENT SYSTEMS LTD",
    "PETRONET LNG LTD", "PFIZER LIMITED", "PG ELECTROPLAST LTD", "PHOENIX MILLS LTD",
    "PI INDUSTRIES LTD", "PIDILITE INDUSTRIES LTD", "PIRAMAL PHARMA LTD",
    "PNB HOUSING FINANCE LTD", "POLY MEDICURE LTD", "POLYCAB INDIA LTD", "POONAWALLA FINCORP LTD",
    "POWER FINANCE CORPORATION", "POWER GRID CORP OF INDIA LTD", "PRAJ INDUSTRIES LTD",
    "PREMIER ENERGIES LTD", "PRESTIGE ESTATES PROJECTS", "PROCTER & GAMBLE HYGIENE",
    "PTC INDUSTRIES LTD", "PUNJAB NATIONAL BANK", "PVR INOX LTD", "RADICO KHAITAN LTD",
    "RAIL VIKAS NIGAM LTD", "RAILTEL CORP OF INDIA LTD", "RAINBOW CHILDREN'S MEDICARE",
    "RAMCO CEMENTS LTD/THE", "RAMKRISHNA FORGINGS LTD", "RASHTRIYA CHEMICALS & FERT",
    "RBL BANK LTD", "REC LTD", "REDINGTON LTD", "RELIANCE INDUSTRIES LIMITED",
    "RELIANCE INFRASTRUCTURE LTD", "RELIANCE POWER LTD", "RHI MAGNESITA INDIA LTD",
    "RITES LTD", "RR KABEL LTD", "SAGILITY LTD", "SAI LIFE SCIENCES LTD", "SAMMAAN CAPITAL LTD",
    "SAMVARDHANA MOTHERSON INTERN", "SAPPHIRE FOODS INDIA LTD", "SARDA ENERGY & MINERALS LTD",
    "SAREGAMA INDIA LTD", "SBFC FINANCE LTD", "SBI CARDS & PAYMENT SERVICES",
    "SBI LIFE INSURANCE CO LTD", "SCHAEFFLER INDIA LTD", "SCHNEIDER ELECTRIC INFRASTRU",
    "SHIPPING CORP OF INDIA LTD", "SHREE CEMENT LTD", "SHRIRAM FINANCE LTD",
    "SHYAM METALICS & ENERGY LTD", "SIEMENS ENERGY INDIA LTD", "SIEMENS LTD",
    "SIGNATUREGLOBAL INDIA LTD", "SJVN LTD", "SOBHA LTD", "SOLAR INDUSTRIES INDIA LTD",
    "SONA BLW PRECISION FORGINGS", "SONATA SOFTWARE LTD", "SRF LTD",
    "STAR HEALTH & ALLIED INSURAN", "STATE BANK OF INDIA", "STEEL AUTHORITY OF INDIA",
    "SUMITOMO CHEMICAL INDIA LTD", "SUN PHARMACEUTICAL INDUS", "SUN TV NETWORK LTD",
    "SUNDARAM FINANCE LTD", "SUNDRAM FASTENERS LTD", "SUPREME INDUSTRIES LTD",
    "SUZLON ENERGY LTD", "SWAN CORP LTD", "SWIGGY LTD", "SYNGENE INTERNATIONAL LTD",
    "SYRMA SGS TECHNOLOGY LTD", "TATA CHEMICALS LTD", "TATA COMMUNICATIONS LTD",
    "TATA CONSULTANCY SVCS LTD", "TATA CONSUMER PRODUCTS LTD", "TATA ELXSI LTD",
    "TATA INVESTMENT CORP LTD", "TATA MOTORS PASSENGER VEHICL", "TATA POWER CO LTD",
    "TATA STEEL LTD", "TATA TECHNOLOGIES CO", "TATA TELESERVICES MAHARASHTR", "TBO TEK LTD",
    "TECH MAHINDRA LTD", "TECHNO ELECTRIC & ENGINEERIN", "TEJAS NETWORKS LTD", "THERMAX LTD",
    "TIMKEN INDIA LTD", "TITAGARH RAIL SYSTEM LTD", "TITAN CO LTD",
    "TORRENT PHARMACEUTICALS LTD", "TORRENT POWER LTD", "TRANSFORMERS & RECTIFIERS IN",
    "TRENT LTD", "TRIDENT LTD", "TRIVENI ENGINEERING & INDUS", "TRIVENI TURBINE LTD",
    "TUBE INVESTMENTS OF INDIA LT", "TVS MOTOR CO LTD", "UCO BANK", "ULTRATECH CEMENT LTD",
    "UNION BANK OF INDIA", "UNITED BREWERIES LTD", "UNITED SPIRITS LTD", "UNO MINDA LTD",
    "UPL LTD", "USHA MARTIN LTD", "UTI ASSET MANAGEMENT CO LTD", "V-GUARD INDUSTRIES LTD",
    "VALOR ESTATE LTD", "VARDHMAN TEXTILES LTD", "VARUN BEVERAGES LTD", "VEDANT FASHIONS LTD",
    "VEDANTA LTD", "VENTIVE HOSPITALITY LTD", "VIJAYA DIAGNOSTIC CENTRE PVT", "VISHAL MEGA MART LTD",
    "VODAFONE IDEA LTD", "VOLTAS LTD", "WAAREE ENERGIES LTD", "WELSPUN CORP LTD",
    "WELSPUN LIVING LTD", "WHIRLPOOL OF INDIA LTD", "WIPRO LTD", "WOCKHARDT LTD", "YES BANK LTD",
    "ZEE ENTERTAINMENT ENTERPRISE", "ZEN TECHNOLOGIES LTD", "ZENSAR TECHNOLOGIES LTD",
    "ZF COMMERCIAL VEHICLE CONTRO", "ZYDUS LIFESCIENCES LTD"
]

# ============================================================================
# ESG KEYWORDS - STRICT INCIDENT (Original - UNCHANGED)
# ============================================================================

ESG_INCIDENT_KEYWORDS = {
    "Environmental": [
        'pollution', 'polluting', 'polluted', 'emission', 'emissions',
        'ngt', 'national green tribunal', 'environment', 'environmental',
        'toxic', 'hazardous', 'contamination', 'contaminated', 'spill',
        'waste', 'dumping', 'effluent', 'deforestation', 'forest',
        'carbon', 'climate', 'green', 'sustainability', 'sustainable',
        'eco', 'ecology', 'cpcb', 'pcb', 'air quality', 'water quality'
    ],
    "Social": [
        'accident', 'death', 'died', 'killed', 'fatal', 'fatality',
        'injury', 'injured', 'worker', 'workers', 'labour', 'labor',
        'strike', 'protest', 'protesting', 'safety', 'unsafe',
        'harassment', 'discrimination', 'child labour', 'human rights',
        'employee', 'employees', 'workplace', 'factory', 'plant',
        'fire', 'explosion', 'blast', 'collapse', 'trapped'
    ],
    "Governance": [
        'sebi', 'fraud', 'scam', 'bribery', 'bribe', 'corruption',
        'insider trading', 'penalty', 'fine', 'fined', 'violation',
        'audit', 'auditor', 'investigation', 'probe', 'inquiry',
        'arrest', 'arrested', 'ed', 'enforcement directorate', 'cbi',
        'money laundering', 'tax evasion', 'default', 'defaulter',
        'misappropriation', 'embezzlement', 'regulatory', 'compliance'
    ]
}

# ============================================================================
# ESG KEYWORDS - BROAD DISCOVERY (NEW - Modern ESG Terminology)
# ============================================================================

ESG_DISCOVERY_KEYWORDS = {
    "Environmental": [
        # Net Zero & Climate
        'net zero', 'net-zero', 'carbon neutral', 'carbon neutrality',
        'decarbonization', 'decarbonisation', 'climate change', 'climate action',
        'climate risk', 'climate target', 'climate commitment',
        # Renewable & Energy
        'renewable', 'renewables', 'renewable energy', 'solar', 'wind energy',
        'clean energy', 'energy transition', 'energy efficiency',
        # Sustainability
        'sustainability report', 'esg report', 'esg rating', 'esg disclosure',
        # Circular Economy & Resources
        'circular economy', 'recycling', 'waste management', 'water scarcity',
        'water management', 'resource efficiency', 'biodiversity',
        # Green Finance
        'green bond', 'green bonds', 'green finance', 'sustainable finance',
        # Emissions Reporting
        'scope 1', 'scope 2', 'scope 3', 'ghg emissions', 'carbon footprint',
        'carbon disclosure', 'cdp', 'sbti', 'science based targets'
    ],
    "Social": [
        # DEI
        'diversity', 'equity', 'inclusion', 'dei', 'd&i',
        'diversity and inclusion', 'gender diversity', 'women leadership',
        'equal opportunity', 'inclusive workplace',
        # Human Rights & Labor
        'labour rights', 'fair wage', 'living wage',
        'modern slavery', 'forced labor', 'forced labour',
        # CSR & Community
        'csr', 'corporate social responsibility', 'community engagement',
        'community development', 'social impact', 'philanthropy',
        # Supply Chain
        'supply chain', 'supplier code', 'responsible sourcing',
        'ethical sourcing', 'supply chain audit',
        # Employee
        'employee wellbeing', 'employee well-being', 'employee welfare',
        'workplace wellness', 'mental health', 'work-life balance',
        'employee engagement', 'talent development', 'upskilling'
    ],
    "Governance": [
        # Board & Leadership
        'board diversity', 'independent director', 'board composition',
        'board oversight', 'board governance', 'corporate governance',
        # Ethics & Compliance
        'ethics', 'ethical', 'code of conduct', 'business ethics',
        'compliance framework', 'regulatory compliance',
        # Transparency
        'transparency', 'disclosure', 'annual report', 'integrated report',
        # Shareholder Rights
        'shareholder rights', 'shareholder engagement', 'shareholder activism',
        'proxy voting', 'agm', 'annual general meeting',
        # Executive Pay
        'executive compensation', 'executive pay', 'ceo pay ratio',
        'remuneration policy', 'incentive alignment',
        # Risk & Controls
        'risk management', 'enterprise risk', 'internal controls',
        'internal audit', 'fiduciary', 'fiduciary duty',
        # Anti-Corruption
        'aml', 'anti-money laundering', 'anti-corruption', 'anti-bribery',
        'whistleblower', 'whistleblowing', 'speak up',
        # Stewardship
        'stewardship', 'stewardship code', 'responsible investment'
    ]
}

# Combined ESG keywords (Incident + Discovery)
ESG_MUST_HAVE = {
    category: ESG_INCIDENT_KEYWORDS[category] + ESG_DISCOVERY_KEYWORDS[category]
    for category in ["Environmental", "Social", "Governance"]
}

# HIGH-SEVERITY INCIDENT KEYWORDS
INCIDENT_SEVERITY_KEYWORDS = {
    "high": [
        'fine', 'penalty', 'penalized', 'penalised', 'court', 'jail', 'prison',
        'order', 'ban', 'banned', 'arrest', 'arrested', 'criminal', 'crime',
        'death', 'deaths', 'fatal', 'fatality', 'killed', 'shutdown',
        'suspension', 'suspended', 'revoked', 'blacklist', 'blacklisted',
        'fraud', 'scam', 'scandal', 'violation', 'breach'
    ],
    "medium": [
        'investigation', 'investigate', 'probe', 'probing', 'notice',
        'inquiry', 'scrutiny', 'warning', 'warned', 'summon', 'summoned',
        'hearing', 'complaint', 'allegation', 'alleged', 'accused'
    ]
}

# Keywords that indicate EXCLUSION (financial news, not ESG)
EXCLUDE_KEYWORDS = [
    'quarterly results', 'q1 results', 'q2 results', 'q3 results', 'q4 results',
    'profit rises', 'profit falls', 'revenue grows', 'revenue up', 'revenue down',
    'stock price', 'share price', 'target price', 'buy rating', 'sell rating',
    'hold rating', 'upgrade', 'downgrade', 'market cap', 'ipo', 'bonus issue',
    'dividend announced', 'dividend declared', 'stock split', 'rights issue',
    'analyst rating', 'brokerage report', 'earnings beat', 'earnings miss',
    'pe ratio', 'eps', 'ebitda', 'margin expansion', 'margin contraction',
    'order book', 'order inflow', 'contract win', 'deal signed',
    'expansion plan', 'capacity addition', 'new plant', 'inaugurates',
    'launches new', 'new product', 'appoints', 'appointment', 'resigns',
    'resignation', 'steps down', 'promoter stake', 'stake sale', 'stake buy'
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sanitize_filename(name: str) -> str:
    """Convert company name to valid folder name."""
    clean = re.sub(r'[<>:"/\\|?*&\']', '', name)
    clean = clean.replace(' ', '_').replace('.', '').replace(',', '')
    return clean[:50]


def get_random_headers() -> Dict:
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }


def generate_article_id(title: str, url: str) -> str:
    content = f"{title}_{url}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def get_company_search_names(company: str) -> List[str]:
    """Get search-friendly names for a company."""
    if company in COMPANY_ALIASES:
        return COMPANY_ALIASES[company]
    
    names = []
    name = company.upper()
    suffixes = [' LTD', ' LIMITED', ' INDIA', ' CORP', ' CO', ' PVT', ' PRIVATE', ' ENTERPRISE', ' ENTERPRISES']
    for suffix in suffixes:
        name = name.replace(suffix, '')
    name = name.strip()
    
    words = name.split()
    if len(words) >= 2:
        names.append(' '.join(words[:2]))
    if len(words) >= 3:
        names.append(' '.join(words[:3]))
    names.append(name)
    
    return list(set(names))


def load_progress() -> Dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"completed": [], "partial": {}, "total": 0, "dropped_old": 0}


def save_progress(progress: Dict):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)


def create_company_folders(company: str) -> str:
    company_folder = sanitize_filename(company)
    company_path = os.path.join(ARTICLES_DIR, company_folder)
    
    for category in ["Environmental", "Social", "Governance"]:
        os.makedirs(os.path.join(company_path, category), exist_ok=True)
    
    return company_path


def parse_article_date(date_str: str) -> Optional[datetime]:
    """Parse article publication date from various formats. Returns naive datetime."""
    if not date_str:
        return None
    
    parsed_dt = None
    
    if HAS_DATEUTIL:
        try:
            parsed_dt = date_parser.parse(date_str, fuzzy=True)
        except:
            pass
    
    if parsed_dt is None:
        # Fallback formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %Z',
            '%a, %d %b %Y %H:%M:%S GMT',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d',
            '%d %b %Y',
            '%B %d, %Y',
            '%d-%m-%Y',
            '%m/%d/%Y'
        ]
        
        for fmt in formats:
            try:
                parsed_dt = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
    
    # Convert to naive datetime (remove timezone info) to avoid comparison errors
    if parsed_dt is not None and parsed_dt.tzinfo is not None:
        parsed_dt = parsed_dt.replace(tzinfo=None)
    
    return parsed_dt


def is_within_time_window(article_date: Optional[datetime], window_days: int = TIME_WINDOW_DAYS) -> bool:
    """Check if article is within the rolling time window (default 5 years)."""
    if article_date is None:
        return True  # Keep articles with unknown dates
    
    try:
        # Ensure we're comparing naive datetimes
        now = datetime.now()
        if article_date.tzinfo is not None:
            article_date = article_date.replace(tzinfo=None)
        
        cutoff_date = now - timedelta(days=window_days)
        return article_date >= cutoff_date
    except:
        return True  # On any error, keep the article


# ============================================================================
# ARTICLE VALIDATOR
# ============================================================================

class ArticleValidator:
    """
    Validates ESG news articles with both strict incident and broad discovery pipelines.
    """
    
    def __init__(self):
        self.esg_keywords = ESG_MUST_HAVE
        self.incident_keywords = ESG_INCIDENT_KEYWORDS
        self.discovery_keywords = ESG_DISCOVERY_KEYWORDS
        self.severity_keywords = INCIDENT_SEVERITY_KEYWORDS
        self.exclude_keywords = EXCLUDE_KEYWORDS
    
    def is_about_company(self, text: str, company: str) -> bool:
        """Check if article is actually about this company."""
        text_lower = text.lower()
        search_names = get_company_search_names(company)
        
        for name in search_names:
            if name.lower() in text_lower:
                return True
        return False
    
    def should_exclude(self, text: str) -> bool:
        """Check if article is financial news (not ESG)."""
        text_lower = text.lower()
        exclude_count = sum(1 for kw in self.exclude_keywords if kw in text_lower)
        return exclude_count >= 2
    
    def get_esg_category(self, text: str) -> Optional[str]:
        """Determine which ESG category the article belongs to."""
        text_lower = text.lower()
        scores = {"Environmental": 0, "Social": 0, "Governance": 0}
        
        for category, keywords in self.esg_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[category] += 1
        
        max_score = max(scores.values())
        if max_score < 2:
            return None
        
        return max(scores, key=scores.get)
    
    def is_incident(self, text: str, category: str) -> Tuple[bool, float]:
        """
        Determine if article describes an ESG incident.
        Returns: (is_incident, severity_score)
        """
        text_lower = text.lower()
        
        # Check high-severity
        high_count = sum(1 for kw in self.severity_keywords['high'] if kw in text_lower)
        if high_count >= 1:
            return True, 3.0
        
        # Check medium-severity
        medium_count = sum(1 for kw in self.severity_keywords['medium'] if kw in text_lower)
        if medium_count >= 1:
            return True, 2.0
        
        # Check strict incident keywords
        incident_count = sum(1 for kw in self.incident_keywords[category] if kw in text_lower)
        if incident_count >= 2:
            return True, 1.5
        
        return False, 1.0
    
    def get_discovery_pipeline(self, text: str, category: str) -> str:
        """Determine which pipeline the article came from."""
        text_lower = text.lower()
        
        incident_count = sum(1 for kw in self.incident_keywords[category] if kw in text_lower)
        discovery_count = sum(1 for kw in self.discovery_keywords[category] if kw in text_lower)
        
        if incident_count > discovery_count:
            return 'strict_incident'
        return 'broad_esg'
    
    def validate_article(self, article: Dict, company: str, target_category: str) -> Tuple[bool, str, Dict]:
        """
        Validate if article is relevant ESG news.
        Returns: (is_valid, reason, metadata)
        """
        text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
        
        metadata = {
            'is_incident': 0,
            'severity': 1.0,
            'discovery_pipeline': 'broad_esg'
        }
        
        if not self.is_about_company(text, company):
            return False, "Not about company", metadata
        
        if self.should_exclude(text):
            return False, "Financial news", metadata
        
        detected_category = self.get_esg_category(text)
        if detected_category is None:
            return False, "No ESG relevance", metadata
        
        if detected_category != target_category:
            return False, f"Category mismatch: {detected_category}", metadata
        
        is_inc, severity = self.is_incident(text, target_category)
        metadata['is_incident'] = 1 if is_inc else 0
        metadata['severity'] = severity
        metadata['discovery_pipeline'] = self.get_discovery_pipeline(text, target_category)
        
        return True, "Valid", metadata


# ============================================================================
# NEWS FETCHER
# ============================================================================

class NewsFetcher:
    """Fetch news from Google News RSS with retry logic."""
    
    def __init__(self):
        self.base_url = "https://news.google.com/rss/search"
        self.session = requests.Session()
        # Set up session with retry
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/rss+xml,application/xml,text/xml,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
        })
    
    def fetch(self, query: str, max_results: int = 15) -> List[Dict]:
        """Fetch news articles with retry logic."""
        articles = []
        
        for attempt in range(3):  # 3 retries
            try:
                params = {
                    'q': query,
                    'hl': 'en-IN',
                    'gl': 'IN',
                    'ceid': 'IN:en'
                }
                
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=20,
                    verify=True
                )
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'xml')
                    items = soup.find_all('item')[:max_results]
                    
                    for item in items:
                        title = item.find('title')
                        description = item.find('description')
                        pub_date = item.find('pubDate')
                        link = item.find('link')
                        source = item.find('source')
                        
                        if title and link:
                            articles.append({
                                'title': title.text.strip() if title else '',
                                'description': self._clean_html(description.text if description else ''),
                                'date': pub_date.text if pub_date else '',
                                'url': link.text if link else '',
                                'source': source.text if source else 'Unknown'
                            })
                    break  # Success, exit retry loop
                    
                elif response.status_code == 429:
                    # Rate limited, wait longer
                    time.sleep(5 * (attempt + 1))
                else:
                    break
                    
            except requests.exceptions.SSLError:
                # Try without SSL verification as fallback
                try:
                    response = self.session.get(
                        self.base_url,
                        params=params,
                        timeout=20,
                        verify=False
                    )
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'xml')
                        items = soup.find_all('item')[:max_results]
                        for item in items:
                            title = item.find('title')
                            link = item.find('link')
                            if title and link:
                                articles.append({
                                    'title': title.text.strip(),
                                    'description': '',
                                    'date': item.find('pubDate').text if item.find('pubDate') else '',
                                    'url': link.text,
                                    'source': 'Unknown'
                                })
                        break
                except:
                    pass
                time.sleep(2)
            except requests.exceptions.ConnectionError:
                time.sleep(3 * (attempt + 1))
            except Exception as e:
                if attempt == 2:  # Last attempt
                    print(f"      Fetch failed: {str(e)[:30]}")
                time.sleep(2)
        
        return articles
    
    def _clean_html(self, text: str) -> str:
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()[:500]


def fetch_article_content(url: str) -> str:
    """Fetch full article content."""
    try:
        response = requests.get(url, headers=get_random_headers(), timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            article = soup.find('article')
            if not article:
                article = soup.find('div', class_=re.compile(r'article|content|story|post|body'))
            
            if article:
                paragraphs = article.find_all('p')
                return ' '.join(p.get_text(strip=True) for p in paragraphs[:15])[:3000]
            
            paragraphs = soup.find_all('p')
            return ' '.join(p.get_text(strip=True) for p in paragraphs[:10])[:2000]
            
    except:
        pass
    return ""


# ============================================================================
# SEARCH QUERIES - INCIDENT + DISCOVERY
# ============================================================================

def get_search_queries(company_name: str, category: str) -> List[str]:
    """
    Generate search queries combining:
    1. Strict incident queries (original - unchanged)
    2. Broad ESG discovery queries (new)
    """
    
    # STRICT INCIDENT QUERIES (UNCHANGED)
    incident_queries = []
    
    if category == "Environmental":
        incident_queries = [
            f'"{company_name}" pollution violation fine',
            f'"{company_name}" NGT environmental penalty',
            f'"{company_name}" toxic waste dumping',
            f'"{company_name}" emission norms breach',
            f'"{company_name}" CPCB notice environment',
        ]
    elif category == "Social":
        incident_queries = [
            f'"{company_name}" worker death accident',
            f'"{company_name}" factory fire explosion',
            f'"{company_name}" labour strike protest',
            f'"{company_name}" workplace safety violation',
            f'"{company_name}" employee harassment case',
        ]
    elif category == "Governance":
        incident_queries = [
            f'"{company_name}" SEBI penalty fraud',
            f'"{company_name}" insider trading violation',
            f'"{company_name}" audit fraud investigation',
            f'"{company_name}" ED CBI probe scam',
            f'"{company_name}" regulatory penalty fine',
        ]
    
    # BROAD ESG DISCOVERY QUERIES (NEW)
    discovery_queries = []
    
    if category == "Environmental":
        discovery_queries = [
            f'"{company_name}" sustainability OR "net zero" OR decarbonization',
            f'"{company_name}" climate OR renewable OR "clean energy"',
            f'"{company_name}" "circular economy" OR biodiversity OR "green bond"',
            f'"{company_name}" "scope 1" OR "scope 2" OR "scope 3" emissions',
            f'"{company_name}" ESG report sustainability disclosure',
            f'"{company_name}" carbon neutral OR carbon footprint',
        ]
    elif category == "Social":
        discovery_queries = [
            f'"{company_name}" diversity OR equity OR inclusion OR DEI',
            f'"{company_name}" CSR OR "corporate social responsibility"',
            f'"{company_name}" "human rights" OR "fair wage" OR "supply chain"',
            f'"{company_name}" "employee wellbeing" OR "community engagement"',
            f'"{company_name}" "women leadership" OR "gender diversity"',
            f'"{company_name}" upskilling OR "talent development" OR training',
        ]
    elif category == "Governance":
        discovery_queries = [
            f'"{company_name}" governance OR "board diversity" OR ethics',
            f'"{company_name}" compliance OR transparency OR disclosure',
            f'"{company_name}" "risk management" OR AML OR "anti-corruption"',
            f'"{company_name}" "executive compensation" OR whistleblower',
            f'"{company_name}" "shareholder rights" OR "annual general meeting"',
            f'"{company_name}" "corporate governance" OR "independent director"',
        ]
    
    # Incident queries first (higher priority), then discovery
    return incident_queries + discovery_queries


# ============================================================================
# MAIN DOWNLOADER
# ============================================================================

class ESGArticleDownloader:
    """
    Download ESG articles with strict validation and broad discovery.
    """
    
    def __init__(self):
        self.fetcher = NewsFetcher()
        self.validator = ArticleValidator()
        self.progress = load_progress()
        self.dropped_old = 0
        os.makedirs(ARTICLES_DIR, exist_ok=True)
    
    def download_for_company(self, company: str) -> Dict[str, int]:
        """Download validated ESG articles for a company."""
        company_path = create_company_folders(company)
        search_names = get_company_search_names(company)
        primary_name = search_names[0] if search_names else company
        
        stats = {"Environmental": 0, "Social": 0, "Governance": 0}
        
        print(f"\n  Searching for: '{primary_name}'")
        
        for category in ["Environmental", "Social", "Governance"]:
            category_path = os.path.join(company_path, category)
            
            existing = [f for f in os.listdir(category_path) if f.endswith('.json')]
            if len(existing) >= MAX_PER_CATEGORY:
                print(f"    {category}: {len(existing)} (already complete)")
                stats[category] = len(existing)
                continue
            
            # Try with 5-year window first, then 10-year if nothing found
            for time_window, window_label in [(TIME_WINDOW_DAYS, "5yr"), (EXTENDED_WINDOW_DAYS, "10yr")]:
                needed = MAX_PER_CATEGORY - len(existing)
                saved = 0
                seen_titles = set()
                
                queries = get_search_queries(primary_name, category)
                
                for query in queries:
                    if saved >= needed:
                        break
                    
                    try:
                        articles = self.fetcher.fetch(query, max_results=10)
                        
                        for article in articles:
                            if saved >= needed:
                                break
                            
                            title_key = article['title'].lower()[:60]
                            if title_key in seen_titles:
                                continue
                            seen_titles.add(title_key)
                            
                            # CHECK TIME WINDOW (5yr or 10yr)
                            article_date = parse_article_date(article.get('date', ''))
                            if not is_within_time_window(article_date, time_window):
                                self.dropped_old += 1
                                continue
                            
                            # Fetch content (skip on error)
                            if article['url']:
                                try:
                                    article['content'] = fetch_article_content(article['url'])
                                except:
                                    article['content'] = ''
                                time.sleep(0.3)
                            else:
                                article['content'] = ''
                            
                            # VALIDATE
                            is_valid, reason, metadata = self.validator.validate_article(article, company, category)
                            
                            if not is_valid:
                                continue
                            
                            # Save
                            article_id = generate_article_id(article['title'], article['url'])
                            filepath = os.path.join(category_path, f"article_{article_id}.json")
                            
                            if os.path.exists(filepath):
                                continue
                            
                            article['company'] = company
                            article['category'] = category
                            article['query_used'] = query
                            article['validated'] = True
                            article['downloaded_at'] = datetime.now().isoformat()
                            article['is_incident'] = metadata['is_incident']
                            article['severity'] = metadata['severity']
                            article['discovery_pipeline'] = metadata['discovery_pipeline']
                            article['time_window'] = window_label  # Track which window was used
                            
                            # Safe date serialization
                            if article_date:
                                try:
                                    article['parsed_date'] = article_date.strftime('%Y-%m-%d')
                                except:
                                    article['parsed_date'] = None
                            else:
                                article['parsed_date'] = None
                            
                            with open(filepath, 'w', encoding='utf-8') as f:
                                json.dump(article, f, indent=2, ensure_ascii=False)
                            
                            saved += 1
                            existing.append(filepath)  # Update existing count
                            inc_marker = "[INC]" if metadata['is_incident'] else "[ESG]"
                            print(f"      {inc_marker}[{window_label}] Saved: {article['title'][:45]}...")
                            
                    except Exception as e:
                        print(f"      Error: {str(e)[:40]}")
                    
                    time.sleep(REQUEST_DELAY)
                
                # If we found articles in 5yr window, don't try 10yr
                if saved > 0:
                    break
                elif time_window == TIME_WINDOW_DAYS:
                    print(f"    {category}: No articles in 5yr, trying 10yr window...")
            
            final_count = len([f for f in os.listdir(category_path) if f.endswith('.json')])
            stats[category] = final_count
            print(f"    {category}: {stats[category]} articles")
        
        return stats
    
    def download_all(self, start_index: int = 0, batch_size: int = 50):
        """Download articles for all companies."""
        
        print("=" * 70)
        print("ESG ARTICLE DOWNLOADER v3 - Strict + Broad Discovery")
        print("=" * 70)
        print(f"Companies: {len(COMPANIES)}")
        print(f"Starting from: #{start_index + 1}")
        print(f"Batch size: {batch_size}")
        print(f"Max per category: {MAX_PER_CATEGORY}")
        print(f"Time window: {TIME_WINDOW_YEARS} years")
        print(f"Output: {ARTICLES_DIR}")
        print("=" * 70)
        print("Legend: [INC] = Incident | [ESG] = ESG Discovery")
        print("=" * 70)
        
        total = 0
        end_index = min(start_index + batch_size, len(COMPANIES))
        
        for idx in range(start_index, end_index):
            company = COMPANIES[idx]
            
            if company in self.progress.get('completed', []):
                print(f"\n[{idx+1}/{len(COMPANIES)}] {company} - SKIPPED")
                continue
            
            print(f"\n[{idx+1}/{len(COMPANIES)}] {company}")
            print("-" * 50)
            
            try:
                stats = self.download_for_company(company)
                count = sum(stats.values())
                total += count
                
                if count >= (MAX_PER_CATEGORY * 3 - 5):
                    self.progress['completed'].append(company)
                else:
                    self.progress['partial'][company] = stats
                
                self.progress['total'] = self.progress.get('total', 0) + count
                self.progress['dropped_old'] = self.progress.get('dropped_old', 0) + self.dropped_old
                save_progress(self.progress)
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
            
            time.sleep(BATCH_DELAY)
        
        print("\n" + "=" * 70)
        print(f"COMPLETE - Downloaded {total} validated articles")
        print(f"Dropped (>5 years old): {self.dropped_old}")
        print("=" * 70)
        
        return total


# ============================================================================
# SUMMARY
# ============================================================================

def generate_summary():
    """Generate summary of downloaded articles."""
    
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    
    if not os.path.exists(ARTICLES_DIR):
        print("No articles found.")
        return
    
    total = 0
    total_incidents = 0
    total_discovery = 0
    stats = []
    
    for folder in os.listdir(ARTICLES_DIR):
        path = os.path.join(ARTICLES_DIR, folder)
        if not os.path.isdir(path):
            continue
        
        s = {"name": folder, "E": 0, "S": 0, "G": 0, "incidents": 0, "discovery": 0}
        for cat in ["Environmental", "Social", "Governance"]:
            cat_path = os.path.join(path, cat)
            if os.path.exists(cat_path):
                articles = [f for f in os.listdir(cat_path) if f.endswith('.json')]
                s[cat[0]] = len(articles)
                
                for art_file in articles:
                    try:
                        with open(os.path.join(cat_path, art_file), 'r', encoding='utf-8') as f:
                            art_data = json.load(f)
                            if art_data.get('is_incident', 0) == 1:
                                s['incidents'] += 1
                            else:
                                s['discovery'] += 1
                    except:
                        pass
        
        s["total"] = s["E"] + s["S"] + s["G"]
        if s["total"] > 0:
            stats.append(s)
            total += s["total"]
            total_incidents += s['incidents']
            total_discovery += s['discovery']
    
    stats.sort(key=lambda x: x['total'], reverse=True)
    
    print(f"Companies with articles: {len(stats)}")
    print(f"Total articles: {total}")
    print(f"  [INC] Incidents: {total_incidents}")
    print(f"  [ESG] Discovery: {total_discovery}")
    
    print(f"\n{'Company':<35} {'E':>3} {'S':>3} {'G':>3} {'Tot':>4} {'Inc':>4} {'Disc':>5}")
    print("-" * 65)
    for s in stats[:30]:
        print(f"{s['name'][:35]:<35} {s['E']:>3} {s['S']:>3} {s['G']:>3} {s['total']:>4} {s['incidents']:>4} {s['discovery']:>5}")
    
    csv_path = os.path.join(BASE_DIR, "download_summary_v3.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Company,Environmental,Social,Governance,Total,Incidents,Discovery\n")
        for s in stats:
            f.write(f"{s['name']},{s['E']},{s['S']},{s['G']},{s['total']},{s['incidents']},{s['discovery']}\n")
    
    print(f"\nSaved: {csv_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--batch', type=int, default=50)
    parser.add_argument('--summary', action='store_true')
    
    args = parser.parse_args()
    
    if args.summary:
        generate_summary()
    else:
        downloader = ESGArticleDownloader()
        downloader.download_all(start_index=args.start, batch_size=args.batch)
        generate_summary()
