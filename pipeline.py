from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from preprocess import clean_text
from blocking import run_blocking
from features import extract_features

def main():
    dataset_dir = Path('/kaggle/input/AMAZOM-ML-CHALLENGE')   # your dataset with train/test tsvs
    output_dir = Path('/kaggle/working/output')

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ... rest of main() unchanged from here (Loading datasets... etc.)


def evaluate_macro_f05(ground_truth_map: dict, prediction_map: dict) -> float:
    scores = []

    for s1_id, y_true in ground_truth_map.items():
        y_pred = prediction_map.get(s1_id, set())

        if not y_true and not y_pred:
            scores.append(1.0)
            continue

        if not y_true and y_pred:
            scores.append(0.0)
            continue

        if y_true and not y_pred:
            scores.append(0.0)
            continue
            
        tp = len(y_true & y_pred)
        fp = len(y_pred - y_true)
        fn = len(y_true - y_pred)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        denom = 0.25 * prec + rec
        f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0
        scores.append(f05)

    return float(np.mean(scores))

def main():
    base_dir = Path(__file__).resolve().parents[3]
    dataset_dir = base_dir / "dataset"
    output_dir = base_dir / "output"
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading datasets...")
    s1_train = pd.read_csv(dataset_dir / "train" / "train_source1.tsv", sep="\t")
    s2_train = pd.read_csv(dataset_dir / "train" / "train_source2.tsv", sep="\t")
    s3_train = pd.read_csv(dataset_dir / "train" / "train_source3.tsv", sep="\t")
    gt_train = pd.read_csv(dataset_dir / "train" / "train_ground_truth.tsv", sep="\t")

    s1_test = pd.read_csv(dataset_dir / "test" / "test_source1.tsv", sep="\t")
    s2_test = pd.read_csv(dataset_dir / "test" / "test_source2.tsv", sep="\t")
    s3_test = pd.read_csv(dataset_dir / "test" / "test_source3.tsv", sep="\t")

    print("Normalizing strings...")
    for df in [s1_train, s2_train, s3_train, s1_test, s2_test, s3_test]:
        df["clean_name"] = df["business_name"].fillna("").apply(lambda x: clean_text(x, is_address=False))
        df["clean_address"] = df["business_address"].fillna("").apply(lambda x: clean_text(x, is_address=True))
        df["country"] = df["country"].fillna("").str.strip()

    s23_train = pd.concat([s2_train, s3_train], ignore_index=True)
    s23_test = pd.concat([s2_test, s3_test], ignore_index=True)

    # Candidate blocking on test data
    print("Running blocking on test data...")
    test_candidates = run_blocking(s1_test, s23_test)
    
    cand_file = output_dir / "candidate_pairs.tsv"

    with open(cand_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in s1_test["entity_id"].tolist():
            cands = ",".join(test_candidates.get(s1_id, []))
            f.write(f"{s1_id}\t{cands}\n")
    print(f"Saved: {cand_file}")

    # Candidate blocking on training data
    print("Running blocking on train data...")
    train_candidates = run_blocking(s1_train, s23_train)

    gt_dict = {}
    for _, row in gt_train.iterrows():
        matches = str(row["matched_entity_ids"]).strip()
        gt_dict[row["source1_entity_id"]] = set(matches.split(",")) if matches and matches != "nan" else set()

    train_rows = []
    for s1_id, cands in train_candidates.items():
        true_set = gt_dict.get(s1_id, set())
        for cand_id in cands:
            train_rows.append({
                "s1_id": s1_id,
                "cand_id": cand_id,
                "label": 1 if cand_id in true_set else 0
            })
            
    df_train_pairs = pd.DataFrame(train_rows)
    s1_dict = s1_train.set_index("entity_id").to_dict("index")
    s23_dict = s23_train.set_index("entity_id").to_dict("index")

    df_train_pairs["clean_name_1"] = df_train_pairs["s1_id"].apply(lambda x: s1_dict[x]["clean_name"])
    df_train_pairs["clean_address_1"] = df_train_pairs["s1_id"].apply(lambda x: s1_dict[x]["clean_address"])
    df_train_pairs["clean_name_2"] = df_train_pairs["cand_id"].apply(lambda x: s23_dict[x]["clean_name"])
    df_train_pairs["clean_address_2"] = df_train_pairs["cand_id"].apply(lambda x: s23_dict[x]["clean_address"])

    print("Extracting features for training pairs...")
    X_train = extract_features(df_train_pairs)
    y_train = df_train_pairs["label"]

    print("Fitting LightGBM classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("Preparing test pairs for inference...")
    test_rows = []

    for s1_id, cands in test_candidates.items():
        for cand_id in cands:
            test_rows.append({"s1_id": s1_id, "cand_id": cand_id})

    test_predictions = {s1_id: [] for s1_id in s1_test["entity_id"].tolist()}
    
    if test_rows:
        df_test_pairs = pd.DataFrame(test_rows)
        s1_test_dict = s1_test.set_index("entity_id").to_dict("index")
        s23_test_dict = s23_test.set_index("entity_id").to_dict("index")

        df_test_pairs["clean_name_1"] = df_test_pairs["s1_id"].apply(lambda x: s1_test_dict[x]["clean_name"])
        df_test_pairs["clean_address_1"] = df_test_pairs["s1_id"].apply(lambda x: s1_test_dict[x]["clean_address"])
        df_test_pairs["clean_name_2"] = df_test_pairs["cand_id"].apply(lambda x: s23_test_dict[x]["clean_name"])
        df_test_pairs["clean_address_2"] = df_test_pairs["cand_id"].apply(lambda x: s23_test_dict[x]["clean_address"])

        print("Extracting features for test pairs...")
        X_test = extract_features(df_test_pairs)
        preds_prob = model.predict_proba(X_test)[:, 1]

        THRESHOLD = 0.70
        for idx, row in df_test_pairs.iterrows():
            if preds_prob[idx] >= THRESHOLD:
                test_predictions[row["s1_id"]].append(row["cand_id"])

    match_file = output_dir / "matching_results.tsv"
    with open(match_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_test["entity_id"].tolist():
            unique_matches = list(dict.fromkeys(test_predictions.get(s1_id, [])))
            f.write(f"{s1_id}\t{','.join(unique_matches)}\n")
            
    print(f"Saved: {match_file}")
    print("Execution complete.")

if __name__ == "__main__":
    main()
