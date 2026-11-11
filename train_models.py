"""
Module 4: Train and compare machine-learning models
Trains Logistic Regression, Decision Tree, and Random Forest,
then evaluates all three using the metrics your report needs.

Run AFTER extract_features.py (needs train_features.csv, test_features.csv)

Install first:
    pip install scikit-learn pandas joblib matplotlib
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

TRAIN_FILE = "train_features.csv"
TEST_FILE = "test_features.csv"


def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # One-hot encode the categorical 'tld' column (e.g. com, xyz, org...)
    # Only keep TLDs seen in training; unseen test TLDs become "other".
    top_tlds = train_df["tld"].value_counts().nlargest(15).index
    joblib.dump(list(top_tlds), "top_tlds.pkl")
    train_df["tld"] = train_df["tld"].where(train_df["tld"].isin(top_tlds), "other")
    test_df["tld"] = test_df["tld"].where(test_df["tld"].isin(top_tlds), "other")

    combined = pd.concat([train_df, test_df], keys=["train", "test"])
    combined = pd.get_dummies(combined, columns=["tld"], prefix="tld")
    train_df = combined.xs("train")
    test_df = combined.xs("test")

    feature_cols = [c for c in train_df.columns if c not in ("url", "label")]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["label"].astype(int)
    X_test = test_df[feature_cols].copy()
    y_test = test_df["label"].astype(int)

    # -1 in domain_age_days / domain_reputation_score means "lookup failed",
    # not a literal negative value. Replace with the median of *known* ages
    # so the model doesn't treat missing data as "very young domain".
    medians = {}
    for col in ["domain_age_days", "domain_reputation_score"]:
        if col in X_train.columns:
            known_median = X_train.loc[X_train[col] != -1, col].median()
            known_median = known_median if not np.isnan(known_median) else 0
            medians[col] = known_median
            X_train[col] = X_train[col].replace(-1, known_median)
            X_test[col] = X_test[col].replace(-1, known_median)
    joblib.dump(medians, "medians.pkl")

    return X_train, X_test, y_train, y_test, feature_cols


def evaluate_model(name, model, X_test, y_test, results, scaled_needed=False, scaler=None):
    X_eval = scaler.transform(X_test) if scaled_needed else X_test
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print(f"\n===== {name} =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(cm)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1-score": f1, "ROC-AUC": auc, "FPR": fpr, "FNR": fnr
    })
    return y_prob


def main():
    X_train, X_test, y_train, y_test, feature_cols = load_data()
    print(f"[*] Training on {len(X_train)} rows, testing on {len(X_test)} rows")
    print(f"[*] Using {len(feature_cols)} features")

    results = []
    roc_data = {}

    # ---- Logistic Regression (needs scaled features) ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    y_prob = evaluate_model("Logistic Regression", logreg, X_test, y_test,
                             results, scaled_needed=True, scaler=scaler)
    roc_data["Logistic Regression"] = y_prob
    joblib.dump(logreg, "model_logreg.pkl")
    joblib.dump(scaler, "scaler.pkl")

    # ---- Decision Tree ----
    dtree = DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42)
    dtree.fit(X_train, y_train)
    y_prob = evaluate_model("Decision Tree", dtree, X_test, y_test, results)
    roc_data["Decision Tree"] = y_prob
    joblib.dump(dtree, "model_dtree.pkl")

    # ---- Random Forest ----
    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_prob = evaluate_model("Random Forest", rf, X_test, y_test, results)
    roc_data["Random Forest"] = y_prob
    joblib.dump(rf, "model_rf.pkl")

    # Save feature column order — needed later by the Flask app
    joblib.dump(feature_cols, "feature_columns.pkl")

    # ---- Comparison table ----
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison.csv", index=False)
    print("\n===== Model Comparison Table =====")
    print(results_df.to_string(index=False))

    best_model = results_df.loc[results_df["F1-score"].idxmax(), "Model"]
    print(f"\n[✓] Best model by F1-score: {best_model}")

    # ---- ROC curve plot ----
    plt.figure(figsize=(7, 6))
    for name, probs in roc_data.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    print("[✓] Saved roc_curves.png")

    print("\n[✓] Saved model_logreg.pkl, model_dtree.pkl, model_rf.pkl, scaler.pkl, "
          "feature_columns.pkl, model_comparison.csv")


if __name__ == "__main__":
    main()