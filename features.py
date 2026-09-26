# features.py calculates string distance and token overlap features over names and addresses

import pandas as pd
from rapidfuzz import fuzz

def jaccard_similarity(s1: str, s2: str) -> float:
    set1, set2 = set(s1.split()), set(s2.split())

    if not set1 or not set2:
        return 0.0
    
    return len(set1 & set2) / len(set1 | set2)

def extract_features(df_pairs: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df_pairs.index)
    
    n1 = df_pairs["clean_name_1"].tolist()
    n2 = df_pairs["clean_name_2"].tolist()
    a1 = df_pairs["clean_address_1"].tolist()
    a2 = df_pairs["clean_address_2"].tolist()
    
    # RapidFuzz name metrics
    features["name_ratio"] = [fuzz.ratio(x, y) / 100.0 for x, y in zip(n1, n2)]
    features["name_partial_ratio"] = [fuzz.partial_ratio(x, y) / 100.0 for x, y in zip(n1, n2)]
    features["name_token_sort"] = [fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(n1, n2)]
    features["name_token_set"] = [fuzz.token_set_ratio(x, y) / 100.0 for x, y in zip(n1, n2)]
    
    # RapidFuzz address metrics
    features["addr_ratio"] = [fuzz.ratio(x, y) / 100.0 for x, y in zip(a1, a2)]
    features["addr_token_sort"] = [fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(a1, a2)]
    features["addr_token_set"] = [fuzz.token_set_ratio(x, y) / 100.0 for x, y in zip(a1, a2)]
    
    # Token Jaccard
    features["name_jaccard"] = [jaccard_similarity(x, y) for x, y in zip(n1, n2)]
    features["addr_jaccard"] = [jaccard_similarity(x, y) for x, y in zip(a1, a2)]
    
    features["name_len_diff"] = [abs(len(x) - len(y)) for x, y in zip(n1, n2)]
    features["addr_len_diff"] = [abs(len(x) - len(y)) for x, y in zip(a1, a2)]
    
    return features