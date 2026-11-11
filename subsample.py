"""
Creates a smaller, balanced subsample from train.csv / test.csv
so WHOIS domain-age lookups finish in a reasonable time.

Target: ~4000 train rows (2000 per class) + ~1000 test rows (500 per class)
"""

import pandas as pd

TRAIN_PER_CLASS = 2000
TEST_PER_CLASS = 500


def subsample(input_file, output_file, per_class):
    df = pd.read_csv(input_file)
    parts = []
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        n = min(per_class, len(subset))
        parts.append(subset.sample(n=n, random_state=42))
    result = pd.concat(parts).sample(frac=1, random_state=42)
    result.to_csv(output_file, index=False)
    print(f"[✓] Saved {output_file}: {len(result)} rows")
    print(result["label"].value_counts())


if __name__ == "__main__":
    subsample("train.csv", "train_sample.csv", TRAIN_PER_CLASS)
    subsample("test.csv", "test_sample.csv", TEST_PER_CLASS)