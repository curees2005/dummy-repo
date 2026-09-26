# preprocess.py standardizes noise

import re

LEGAL_SUFFIXES = {
    r"\bpvt\b\.?": "private",
    r"\bltd\b\.?": "limited",
    r"\bcorp\b\.?": "corporation",
    r"\binc\b\.?": "incorporated",
    r"\bco\b\.?": "company",
    r"\bllc\b\.?": "limited liability company",
    r"\bllp\b\.?": "limited liability partnership",
    r"\bprop\b\.?": "proprietor",
    r"\bsarl\b\.?": "societe a responsabilite limitee",
    r"\bsas\b\.?": "societe par actions simplifiee",
    r"\bsa\b\.?": "societe anonyme",
    r"\bsci\b\.?": "societe civile immobiliere",
    r"\beurl\b\.?": "entreprise unipersonnelle a responsabilite limitee"
}

ADDRESS_ABBRS = {
    r"\brd\b\.?": "road",
    r"\bst\b\.?": "street",
    r"\bave\b\.?": "avenue",
    r"\bblvd\b\.?": "boulevard",
    r"\bln\b\.?": "lane",
    r"\bdr\b\.?": "drive",
    r"\bflr\b\.?": "floor",
    r"\bste\b\.?": "suite",
    r"\bapt\b\.?": "apartment",
    r"\bopp\b\.?": "opposite",
    r"\bnr\b\.?": "near",
    r"\bhwy\b\.?": "highway",
    r"\bclg\b\.?": "colony",
    r"\bmkt\b\.?": "market",
    r"\bnagar\b\.?": "nagar",
    r"\br\b\.?": "rue",
    r"\bbd\b\.?": "boulevard",
    r"\bav\b\.?": "avenue",
    r"\ball\b\.?": "allee",
    r"\bche\b\.?": "chemin"
}

def clean_text(text: str, is_address: bool = False) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
        
    text = text.lower()
    
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    
    replacements = ADDRESS_ABBRS if is_address else LEGAL_SUFFIXES
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
        
    text = re.sub(r"[^\w\s]", " ", text)
    
    return re.sub(r"\s+", " ", text).strip()