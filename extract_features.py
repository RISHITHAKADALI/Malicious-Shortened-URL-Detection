"""
Module 3: Feature Extraction
Converts each raw URL into numeric features for the ML models.

Run AFTER subsample.py (needs train_sample.csv and test_sample.csv)

Install first:
    pip install python-whois tldextract pandas
"""

import re
import time
import socket
import tldextract
import pandas as pd
import whois
from urllib.parse import urlparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Cap each individual WHOIS network call so one unresponsive server
# can't stall the whole run.
socket.setdefaulttimeout(6)

WHOIS_WORKERS = 20  # how many WHOIS lookups to run at the same time

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorte.st", "tiny.cc",
    "bl.ink", "soo.gd", "s2r.co", "clck.ru", "v.gd"
}

SUSPICIOUS_WORDS = [
    "login", "verify", "account", "update", "secure", "banking",
    "confirm", "signin", "webscr", "password", "free", "gift",
    "prize", "win", "lottery", "claim", "urgent", "suspend"
]

_whois_cache = {}


def lookup_single_domain(domain):
    """One WHOIS lookup. Returns (domain, age_in_days_or_-1)."""
    try:
        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation:
            return domain, (datetime.now() - creation).days
    except Exception:
        pass
    return domain, -1


def prefetch_whois(domains, workers=WHOIS_WORKERS):
    """Looks up many domains in parallel and fills the cache."""
    domains = [d for d in set(domains) if d not in _whois_cache]
    total = len(domains)
    if total == 0:
        return

    print(f"[*] Looking up WHOIS for {total} unique domains "
          f"using {workers} parallel workers...")
    done = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(lookup_single_domain, d): d for d in domains}
        for future in as_completed(futures):
            domain, age = future.result()
            _whois_cache[domain] = age
            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - start
                print(f"    {done}/{total} domains looked up  ({round(elapsed, 1)}s elapsed)")

    print(f"[✓] WHOIS prefetch done in {round((time.time() - start) / 60, 1)} minutes")


def get_domain_age_days(domain):
    return _whois_cache.get(domain, -1)


def extract_features(url, do_whois=True):
    features = {}
    parsed = urlparse(url)
    domain_full = parsed.netloc.lower()
    ext = tldextract.extract(url)
    registered_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    # ---- Lexical features ----
    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_special_chars"] = len(re.findall(r"[^a-zA-Z0-9]", url))
    features["num_subdirs"] = parsed.path.count("/")
    features["num_query_params"] = len(parsed.query.split("&")) if parsed.query else 0
    features["has_at_symbol"] = int("@" in url)
    features["has_ip_address"] = int(bool(re.match(
        r"^(\d{1,3}\.){3}\d{1,3}$", domain_full.split(":")[0]
    )))
    features["num_suspicious_words"] = sum(
        1 for w in SUSPICIOUS_WORDS if w in url.lower()
    )

    # ---- Domain features ----
    features["domain_length"] = len(domain_full)
    features["tld"] = ext.suffix
    features["uses_https"] = int(parsed.scheme == "https")
    features["domain_age_days"] = get_domain_age_days(registered_domain) if do_whois else -1

    # Placeholder - see note in project report re: reputation scoring
    # (VirusTotal free-tier rate limits make this infeasible at full dataset scale)
    features["domain_reputation_score"] = -1

    # ---- Shortened URL features ----
    is_shortened = int(registered_domain in SHORTENER_DOMAINS)
    features["is_shortened"] = is_shortened
    features["shortened_code_length"] = (
        len(parsed.path.strip("/")) if is_shortened else 0
    )

    return features


def process_file(input_file, output_file, do_whois=True):
    print(f"\n[*] Processing {input_file} ...")
    df = pd.read_csv(input_file)

    if do_whois:
        domains = [
            (lambda e: f"{e.domain}.{e.suffix}" if e.suffix else e.domain)(tldextract.extract(u))
            for u in df["url"]
        ]
        prefetch_whois(domains)

    rows = []
    total = len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        feats = extract_features(row.url, do_whois=do_whois)
        feats["url"] = row.url
        feats["label"] = row.label
        rows.append(feats)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"    {i + 1}/{total} feature rows built")

    result_df = pd.DataFrame(rows)
    cols = ["url", "label"] + [c for c in result_df.columns if c not in ("url", "label")]
    result_df = result_df[cols]

    result_df.to_csv(output_file, index=False)
    print(f"[✓] Saved {output_file} with {len(result_df)} rows, {len(cols) - 2} features")


if __name__ == "__main__":
    DO_WHOIS = True

    start = time.time()
    process_file("train_sample.csv", "train_features.csv", do_whois=DO_WHOIS)
    process_file("test_sample.csv", "test_features.csv", do_whois=DO_WHOIS)
    print(f"\n[✓] Done in {round((time.time() - start) / 60, 1)} minutes")