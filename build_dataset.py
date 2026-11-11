"""
Module 1: Dataset Collection
Downloads real benign + malicious URLs and merges them into urls.csv
"""

import csv
import io
import zipfile
import random
import requests
import pandas as pd

TRANCO_ZIP_URL = "https://tranco-list.eu/top-1m.csv.zip"
URLHAUS_CSV_URL = "https://urlhaus.abuse.ch/downloads/csv_recent/"
OPENPHISH_URL = "https://openphish.com/feed.txt"

TARGET_PER_CLASS = 25000

def get_benign_urls(n):
    print("[*] Downloading Tranco top sites list...")
    resp = requests.get(TRANCO_ZIP_URL, timeout=60)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    inner_name = z.namelist()[0]
    with z.open(inner_name) as f:
        df = pd.read_csv(f, header=None, names=["rank", "domain"])
    df = df.head(min(len(df), n * 5))
    sample = df.sample(n=min(n, len(df)), random_state=42)
    urls = ["https://" + d for d in sample["domain"].tolist()]
    print(f"[+] Collected {len(urls)} benign URLs from Tranco")
    return urls

def get_malicious_urls(n):
    urls = set()
    try:
        print("[*] Downloading URLhaus recent malicious URLs...")
        resp = requests.get(URLHAUS_CSV_URL, timeout=60)
        resp.raise_for_status()
        lines = [l for l in resp.text.splitlines() if l and not l.startswith("#")]
        reader = csv.reader(lines)
        for row in reader:
            if len(row) > 2:
                url = row[2].strip('"')
                if url.startswith("http"):
                    urls.add(url)
        print(f"[+] Collected {len(urls)} URLs so far from URLhaus")
    except Exception as e:
        print(f"[!] URLhaus download failed: {e}")

    try:
        print("[*] Downloading OpenPhish feed...")
        resp = requests.get(OPENPHISH_URL, timeout=60)
        resp.raise_for_status()
        op_urls = [l.strip() for l in resp.text.splitlines() if l.strip()]
        urls.update(op_urls)
        print(f"[+] Total malicious URLs after OpenPhish: {len(urls)}")
    except Exception as e:
        print(f"[!] OpenPhish download failed: {e}")

    urls = list(urls)
    random.shuffle(urls)

    if len(urls) < n:
        print(f"[!] Only found {len(urls)} malicious URLs (target was {n}).")
    return urls[:n]

def main():
    benign = get_benign_urls(TARGET_PER_CLASS)
    malicious = get_malicious_urls(TARGET_PER_CLASS)
    rows = [(u, "benign") for u in benign] + [(u, "malicious") for u in malicious]
    random.shuffle(rows)
    df = pd.DataFrame(rows, columns=["url", "label"])
    df.drop_duplicates(subset="url", inplace=True)
    df.to_csv("urls.csv", index=False)
    print(f"\n[✓] Saved urls.csv with {len(df)} rows")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()