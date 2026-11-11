"""
Fixes the benign URL data quality issue: adds realistic paths/query
params to benign domains so the model can't trivially separate classes
based on "has a path or not".

Run this BEFORE preprocess.py (it rewrites urls.csv).
If you already ran preprocess.py / extract_features.py, re-run those
two scripts again after this one.
"""

import random
import pandas as pd

random.seed(42)

# A realistic mix of path patterns seen on real benign websites.
PATH_TEMPLATES = [
    "/",
    "/about",
    "/about-us",
    "/contact",
    "/contact-us",
    "/login",
    "/signin",
    "/account",
    "/account/settings",
    "/search?q={word}",
    "/search?query={word}&page={num}",
    "/products/{word}",
    "/product/{num}",
    "/category/{word}",
    "/blog/{word}",
    "/blog/2024/{word}-guide",
    "/news/{word}",
    "/article/{num}",
    "/help",
    "/help/faq",
    "/support",
    "/terms",
    "/privacy",
    "/privacy-policy",
    "/careers",
    "/pricing",
    "/downloads/{word}",
    "/user/{num}",
    "/profile/{num}",
    "/cart",
    "/checkout",
    "/api/v1/{word}",
    "/docs/{word}",
    "/{word}/index.html",
]

WORDS = [
    "guide", "review", "overview", "update", "summary", "report",
    "info", "details", "notes", "features", "services", "solutions",
    "help", "faq", "team", "story", "insights", "resources"
]


def random_path():
    template = random.choice(PATH_TEMPLATES)
    return template.format(
        word=random.choice(WORDS),
        num=random.randint(1, 9999)
    )


def main():
    df = pd.read_csv("urls.csv")

    benign_mask = df["label"] == "benign"
    n_benign = benign_mask.sum()
    print(f"[*] Adding realistic paths to {n_benign} benign URLs...")

    def add_path(url):
        url = url.rstrip("/")
        return url + random_path()

    df.loc[benign_mask, "url"] = df.loc[benign_mask, "url"].apply(add_path)

    df.to_csv("urls.csv", index=False)
    print("[✓] Updated urls.csv with realistic benign URL paths")
    print(df["label"].value_counts())
    print("\nSample benign URLs now:")
    print(df[df["label"] == "benign"]["url"].sample(5, random_state=1).to_string(index=False))


if __name__ == "__main__":
    main()