
"""
Module 2 (part 1): Balance the dataset
Trims the majority class down so benign and malicious counts match.
"""
 
import pandas as pd
 
df = pd.read_csv("urls.csv")
 
benign = df[df["label"] == "benign"]
malicious = df[df["label"] == "malicious"]
 
n = min(len(benign), len(malicious))
print(f"[*] Balancing to {n} rows per class...")
 
benign_sample = benign.sample(n=n, random_state=42)
malicious_sample = malicious.sample(n=n, random_state=42)
 
balanced = pd.concat([benign_sample, malicious_sample]).sample(frac=1, random_state=42)
balanced.to_csv("urls.csv", index=False)
 
print(f"[✓] Saved balanced urls.csv with {len(balanced)} rows")
print(balanced["label"].value_counts())