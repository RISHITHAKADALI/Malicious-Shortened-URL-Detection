"""
Module 2: Data Preprocessing
- Removes duplicates and empty/invalid rows
- Standardizes labels to 0 (benign) / 1 (malicious)
- Splits into train/test sets while avoiding near-duplicate leakage
"""

import re
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split

INPUT_FILE = "urls.csv"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"


def is_valid_url(url):
    """Basic structural check - must have a scheme and a netloc (domain)."""
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme) and bool(parsed.netloc)
    except Exception:
        return False


def get_root_domain(url):
    """
    Used to prevent near-duplicate leakage between train/test.
    e.g. https://amazon.com/page1 and https://amazon.com/page2
    should not end up split across train and test.
    """
    try:
        netloc = urlparse(url).netloc.lower()
        netloc = re.sub(r"^www\.", "", netloc)
        return netloc
    except Exception:
        return url


def main():
    print("[*] Loading raw dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"    Loaded {len(df)} rows")

    # 1. Remove empty values
    before = len(df)
    df.dropna(subset=["url", "label"], inplace=True)
    df = df[df["url"].str.strip() != ""]
    print(f"[*] Removed {before - len(df)} empty rows")

    # 2. Standardize labels -> 0 = benign, 1 = malicious
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    label_map = {"benign": 0, "malicious": 1, "0": 0, "1": 1}
    df["label"] = df["label"].map(label_map)
    before = len(df)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)
    print(f"[*] Removed {before - len(df)} rows with unrecognized labels")

    # 3. Remove invalid URLs (structurally broken)
    before = len(df)
    df = df[df["url"].apply(is_valid_url)]
    print(f"[*] Removed {before - len(df)} structurally invalid URLs")

    # 4. Remove exact duplicate URLs
    before = len(df)
    df.drop_duplicates(subset="url", inplace=True)
    print(f"[*] Removed {before - len(df)} exact duplicate URLs")

    # 5. Check class balance
    print("\n[*] Class balance after cleaning:")
    print(df["label"].value_counts())

    # 6. Split by root domain so the same domain never appears in
    #    both train and test (prevents leakage / inflated accuracy)
    df["root_domain"] = df["url"].apply(get_root_domain)
    unique_domains = df["root_domain"].unique()

    train_domains, test_domains = train_test_split(
        unique_domains, test_size=0.2, random_state=42
    )

    train_df = df[df["root_domain"].isin(train_domains)].drop(columns=["root_domain"])
    test_df = df[df["root_domain"].isin(test_domains)].drop(columns=["root_domain"])

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"\n[✓] Saved {TRAIN_FILE}: {len(train_df)} rows")
    print(train_df["label"].value_counts())
    print(f"\n[✓] Saved {TEST_FILE}: {len(test_df)} rows")
    print(test_df["label"].value_counts())


if __name__ == "__main__":
    main()