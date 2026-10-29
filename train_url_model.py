"""PhishGuard - URL-Only Random Forest Model Training and Evaluation.

This script trains and evaluates a separate, standalone machine-learning model
trained STRICTLY on the 8 verified URL-derived lexical features from the UCI dataset:
1. having_ip_address
2. url_length
3. shortining_service
4. having_at_symbol
5. double_slash_redirecting
6. prefix_suffix
7. having_sub_domain
8. https_token

All other 22 dataset features (requiring network probes, live DNS queries,
SSL certificate verification, HTML DOM parsing, or external APIs) are excluded.

Outputs:
- models/phishing_url_only_model.joblib
- models/url_only_model_metadata.joblib
"""

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "phishing_websites.csv"
MODELS_DIR = BASE_DIR / "models"
URL_MODEL_FILE = MODELS_DIR / "phishing_url_only_model.joblib"
URL_METADATA_FILE = MODELS_DIR / "url_only_model_metadata.joblib"

# Configuration
TARGET_COL = "result"
EXPECTED_LABELS = {-1, 1}
LABEL_NAMES = {-1: "Phishing (-1)", 1: "Legitimate (1)"}
RANDOM_STATE = 42
N_ESTIMATORS = 200
TEST_SIZE = 0.20

# The 8 verified lexical URL features and their scientific justification
VERIFIED_URL_FEATURES = [
    "having_ip_address",
    "url_length",
    "shortining_service",
    "having_at_symbol",
    "double_slash_redirecting",
    "prefix_suffix",
    "having_sub_domain",
    "https_token"
]

FEATURE_JUSTIFICATIONS = {
    "having_ip_address": "Parsed directly from URL host using IP address syntax rules.",
    "url_length": "Character count of the raw URL string (<54, 54-75, >75).",
    "shortining_service": "Hostname matched against documented URL shortener domains.",
    "having_at_symbol": "Character search for '@' in the raw URL string.",
    "double_slash_redirecting": "Position check for '//' inside the URL path beyond scheme.",
    "prefix_suffix": "Character check for '-' (hyphen) in the domain part.",
    "having_sub_domain": "Dot count in domain part after removing ccTLD/TLD/www.",
    "https_token": "Token search for 'https' embedded in the hostname part."
}


def load_and_validate_url_dataset():
    """Load dataset, filter to the 8 URL features, and validate integrity."""
    print("=" * 80)
    print("PHISHGUARD - URL-ONLY MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 80)
    print(f"\n[1/5] Loading and filtering dataset from: {CSV_PATH.name}")

    if not CSV_PATH.exists():
        print(f"[FATAL ERROR] Dataset file not found at: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"  -> Total instances in dataset : {len(df):,}")
    print(f"  -> Total dataset columns      : {len(df.columns)}")

    # Target column validation
    if TARGET_COL not in df.columns:
        print(f"[FATAL ERROR] Target column '{TARGET_COL}' missing!")
        sys.exit(1)

    unique_labels = set(df[TARGET_COL].unique())
    if unique_labels != EXPECTED_LABELS:
        print(f"[FATAL ERROR] Unexpected labels {unique_labels}, expected {EXPECTED_LABELS}")
        sys.exit(1)

    # Verify all 8 features exist in the dataset
    missing_cols = [c for c in VERIFIED_URL_FEATURES if c not in df.columns]
    if missing_cols:
        print(f"[FATAL ERROR] Required URL features missing from dataset: {missing_cols}")
        sys.exit(1)

    X_url = df[VERIFIED_URL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    print(f"\n  Selected {len(VERIFIED_URL_FEATURES)} Verified URL-Only Features:")
    for i, feat in enumerate(VERIFIED_URL_FEATURES, 1):
        print(f"    {i}. {feat:<26} : {FEATURE_JUSTIFICATIONS[feat]}")

    print(f"\n  -> Target classes: -1 = Phishing ({sum(y == -1):,}), 1 = Legitimate ({sum(y == 1):,})")
    print(f"  -> Excluded 22 features requiring network, DNS, SSL, HTML DOM, or third-party APIs.")

    return df, X_url, y


def analyze_feature_patterns(df, X_url, y):
    """Analyze patterns and conflicts across the 8 URL features."""
    print("\n" + "=" * 80)
    print("[2/5] URL FEATURE SPACE ANALYSIS & PATTERN CONFLICTS")
    print("=" * 80)

    total_rows = len(df)
    unique_patterns = X_url.drop_duplicates().shape[0]
    duplicate_rows = X_url.duplicated().sum()

    # Conflicting patterns: identical 8 features but different target label
    grouped = df.groupby(VERIFIED_URL_FEATURES)[TARGET_COL].nunique()
    conflicting_patterns = (grouped > 1).sum()

    print(f"  - Total dataset rows                   : {total_rows:,}")
    print(f"  - Unique 8-feature patterns            : {unique_patterns}")
    print(f"  - Duplicate rows sharing 8 features    : {duplicate_rows:,} ({duplicate_rows / total_rows * 100:.2f}%)")
    print(f"  - Conflicting patterns (both labels)   : {conflicting_patterns} out of {unique_patterns} ({conflicting_patterns / unique_patterns * 100:.1f}%)")

    print("\n  [Key Scientific Insight]:")
    print(f"  Across all 11,055 websites, there are only {unique_patterns} distinct combinations of these 8 URL features.")
    print(f"  In {conflicting_patterns} of these patterns, both phishing and legitimate sites share the EXACT same URL traits.")
    print("  This mathematically establishes the upper-bound limitation of purely lexical URL detection:")
    print("  Some phishing sites deliberately use clean URL syntax, and some benign sites use complex subdomains/lengths.")


def train_and_evaluate_standard_split(X_url, y):
    """Evaluate URL-only model using standard 80:20 stratified split."""
    print("\n" + "=" * 80)
    print("[3/5] EXPERIMENT 1: STANDARD STRATIFIED 80:20 SPLIT")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X_url,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"  -> Training split : {X_train.shape[0]:,} samples (80%)")
    print(f"  -> Testing split  : {X_test.shape[0]:,} samples (20%)")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print(f"  Training URL-Only RandomForestClassifier ({N_ESTIMATORS} trees)...")
    model.fit(X_train, y_train)

    # Evaluate on held-out test split
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec_phish = float(precision_score(y_test, y_pred, pos_label=-1))
    rec_phish = float(recall_score(y_test, y_pred, pos_label=-1))
    f1_phish = float(f1_score(y_test, y_pred, pos_label=-1))

    print("\n" + "-" * 80)
    print("STANDARD 80:20 SPLIT RESULTS (8 URL FEATURES):")
    print("-" * 80)
    print(f"  Overall Accuracy     : {acc * 100:.2f}% ({acc:.4f})")
    print(f"  Phishing Precision   : {prec_phish * 100:.2f}%  (When model predicts phishing, it is correct {prec_phish * 100:.1f}%)")
    print(f"  Phishing Recall      : {rec_phish * 100:.2f}%  (Model caught {rec_phish * 100:.1f}% of all actual phishing URLs)")
    print(f"  Phishing F1-Score    : {f1_phish * 100:.2f}%  (Harmonic mean of precision and recall)")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    fp, tn = int(cm[1, 0]), int(cm[1, 1])

    print("\n  Confusion Matrix (Rows = Actual, Columns = Predicted):")
    print("  -------------------------------------------------------")
    print(f"                           Pred Phishing (-1)   Pred Legitimate (1)")
    print(f"  Actual Phishing (-1)    : {tp:6d} (TP)        {fn:6d} (FN)")
    print(f"  Actual Legitimate (1)   : {fp:6d} (FP)        {tn:6d} (TN)")
    print("  -------------------------------------------------------")

    print("\n  Classification Report:")
    target_names = ["Phishing (-1)", "Legitimate (1)"]
    print(classification_report(y_test, y_pred, labels=[-1, 1], target_names=target_names, digits=4))

    metrics = {
        "accuracy": acc,
        "phishing_precision": prec_phish,
        "phishing_recall": rec_phish,
        "phishing_f1": f1_phish,
        "confusion_matrix": {
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn
        }
    }

    return model, metrics


def train_and_evaluate_grouped_split(df, X_url, y):
    """Evaluate URL-only model using 5-Fold StratifiedGroupKFold to prevent pattern leakage."""
    print("=" * 80)
    print("[4/5] EXPERIMENT 2: FAIR GROUPED EVALUATION (STRATIFIED GROUP K-FOLD)")
    print("=" * 80)

    groups = df.groupby(VERIFIED_URL_FEATURES).ngroup()
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_accs = []
    fold_precs = []
    fold_recs = []
    fold_f1s = []

    print(f"  Running {n_splits}-fold grouped cross-validation across {groups.nunique()} unique feature groups...")
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_url, y, groups), 1):
        X_tr, y_tr = X_url.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X_url.iloc[test_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        fold_accs.append(accuracy_score(y_te, preds))
        fold_precs.append(precision_score(y_te, preds, pos_label=-1, zero_division=0))
        fold_recs.append(recall_score(y_te, preds, pos_label=-1, zero_division=0))
        fold_f1s.append(f1_score(y_te, preds, pos_label=-1, zero_division=0))

        print(f"    Fold {fold}: Acc={fold_accs[-1]*100:.2f}% | Phishing Rec={fold_recs[-1]*100:.2f}% | Phishing F1={fold_f1s[-1]*100:.2f}%")

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    mean_prec = float(np.mean(fold_precs))
    mean_rec = float(np.mean(fold_recs))
    mean_f1 = float(np.mean(fold_f1s))

    print("\n" + "-" * 80)
    print(f"GROUPED {n_splits}-FOLD CV SUMMARY (TRUE GENERALIZATION ON UNSEEN URLS):")
    print("-" * 80)
    print(f"  Mean Accuracy        : {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)")
    print(f"  Mean Phishing Prec   : {mean_prec * 100:.2f}%")
    print(f"  Mean Phishing Recall : {mean_rec * 100:.2f}%")
    print(f"  Mean Phishing F1     : {mean_f1 * 100:.2f}%")

    grouped_metrics = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_phishing_precision": mean_prec,
        "mean_phishing_recall": mean_rec,
        "mean_phishing_f1": mean_f1
    }

    return grouped_metrics


def save_url_model_artifacts(model, std_metrics, grouped_metrics):
    """Save the fitted 8-feature URL model and metadata without touching the 30-feature model."""
    print("\n" + "=" * 80)
    print("[5/5] SAVING URL-ONLY MODEL ARTIFACTS")
    print("=" * 80)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save model
    joblib.dump(model, URL_MODEL_FILE)
    print(f"  -> Model saved to: {URL_MODEL_FILE.relative_to(BASE_DIR)}")

    # 2. Save comprehensive metadata
    metadata = {
        "model": model,
        "model_name": "PhishGuard URL-Only Random Forest Classifier",
        "feature_names": VERIFIED_URL_FEATURES,
        "feature_count": len(VERIFIED_URL_FEATURES),
        "target_column": TARGET_COL,
        "labels": LABEL_NAMES,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "metrics_standard_split": std_metrics,
        "metrics_grouped_cv": grouped_metrics,
        "limitations": (
            "Model is trained strictly on 8 lexical URL features. While phishing recall is strong (~80-84%), "
            "overall accuracy is ~71-73% (compared to 95-97% for the 30-feature model) because benign and "
            "phishing websites often share identical URL syntax. Content and DOM verification are necessary "
            "for definitive classification."
        )
    }

    joblib.dump(metadata, URL_METADATA_FILE)
    print(f"  -> Metadata saved to: {URL_METADATA_FILE.relative_to(BASE_DIR)}")

    # Verify saved artifact
    loaded = joblib.load(URL_METADATA_FILE)
    assert len(loaded["feature_names"]) == 8
    print("  -> Verification passed: Loaded metadata with 8 features.")


if __name__ == "__main__":
    df, X_url, y = load_and_validate_url_dataset()
    analyze_feature_patterns(df, X_url, y)
    model, std_metrics = train_and_evaluate_standard_split(X_url, y)
    grouped_metrics = train_and_evaluate_grouped_split(df, X_url, y)
    save_url_model_artifacts(model, std_metrics, grouped_metrics)
    print("\n[SUCCESS] URL-Only Model training and evaluation completed successfully!")
