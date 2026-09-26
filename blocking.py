# blocking.py implements multi-rule blocking strategy

from collections import defaultdict
import pandas as pd
from rapidfuzz import fuzz

IGNORED_TOKENS = {
    "the", "a", "an", "new", "shree", "sri", "hotel", "restaurant", 
    "store", "shop", "enterprise", "enterprises", "solutions", "services",
    "company", "trading", "industries", "agency", "center", "centre", "corp",
    "ltd", "pvt", "llc", "inc", "co", "road", "street", "avenue", "lane", "box"
}

def extract_index_keys(name: str, address: str) -> list[str]:
    keys = []
    name_tokens = [t for t in name.split() if len(t) >= 3 and t not in IGNORED_TOKENS]

    if name_tokens:
        keys.append("N1_" + name_tokens[0])
        if len(name_tokens) > 1:
            keys.append("N2_" + name_tokens[1])
            
    addr_tokens = [t for t in address.split() if len(t) >= 4 and t not in IGNORED_TOKENS]
    if addr_tokens:
        keys.append("A1_" + addr_tokens[0])
        
    return keys

# Scales to millions of records using bounded inverted index retrieval
def run_blocking(df_s1: pd.DataFrame, df_s23: pd.DataFrame, max_cands_per_query: int = 15) -> dict:
    candidates = defaultdict(set)
    all_s1_ids = df_s1["entity_id"].tolist()
    
    countries = df_s1["country"].unique()
    
    for c in countries:
        s1_c = df_s1[df_s1["country"] == c]
        s23_c = df_s23[df_s23["country"] == c]
        
        if s23_c.empty or s1_c.empty:
            continue
            
        print(f"[{c}] Indexing {len(s23_c)} reference records...")
        
        # Build bounded inverted index
        inv_index = defaultdict(list)
        s23_meta = {}
        
        for e_id, name, addr in zip(s23_c["entity_id"], s23_c["clean_name"], s23_c["clean_address"]):
            s23_meta[e_id] = (name, addr)
            keys = extract_index_keys(name, addr)

            for k in keys:
                if len(inv_index[k]) < 100:
                    inv_index[k].append(e_id)

        print(f"[{c}] Querying index for {len(s1_c)} Source 1 records...")
        
        # Query each S1 record against posting lists
        for s1_id, s1_name, s1_addr in zip(s1_c["entity_id"], s1_c["clean_name"], s1_c["clean_address"]):
            keys = extract_index_keys(s1_name, s1_addr)
            pool = set()

            for k in keys:
                for match_id in inv_index.get(k, []):
                    pool.add(match_id)
            
            if not pool:
                continue
                
            if len(pool) <= max_cands_per_query:
                candidates[s1_id].update(pool)
            else:
                scored = []
                for cand_id in pool:
                    c_name, c_addr = s23_meta[cand_id]
                    score = fuzz.token_sort_ratio(s1_name, c_name) * 0.7 + fuzz.ratio(s1_addr, c_addr) * 0.3
                    scored.append((score, cand_id))
                
                # Keep top K candidates
                scored.sort(key=lambda x: x[0], reverse=True)

                for score, cand_id in scored[:max_cands_per_query]:
                    if score >= 35.0: 
                        candidates[s1_id].add(cand_id)

    return {s1_id: list(candidates[s1_id]) for s1_id in all_s1_ids}