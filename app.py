
import logging
import ipaddress
import re
from urllib.parse import urlparse

import joblib
import pandas as pd
import tldextract

from flask import Flask, render_template, request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = "model_rf.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
MEDIANS_PATH = "medians.pkl"
TOP_TLDS_PATH = "top_tlds.pkl"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
medians = joblib.load(MEDIANS_PATH)
top_tlds = joblib.load(TOP_TLDS_PATH)

# --------------------------------------------------
# TRUSTED DOMAIN OVERRIDE
# --------------------------------------------------

TRUSTED_DOMAINS = {
    "google.com",
    "youtube.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "wikipedia.org",
    "github.com",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
}

# --------------------------------------------------
# SUSPICIOUS KEYWORDS
# --------------------------------------------------

SUSPICIOUS_WORDS = [
    "login",
    "verify",
    "account",
    "secure",
    "update",
    "password",
    "banking",
    "confirm",
    "signin",
    "wallet",
    "payment",
    "free",
    "bonus",
    "suspend",
]

SHORTENED_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "cutt.ly",
    "shorte.st",
    "tiny.cc",
    "bl.ink",
    "soo.gd",
    "s2r.co",
    "clck.ru",
    "v.gd",
}

# --------------------------------------------------
# URL VALIDATION
# --------------------------------------------------

def normalize_url(url):
    url = url.strip()

    if not url:
        raise ValueError("Please enter a URL.")

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)

    if not parsed.hostname:
        raise ValueError("Please enter a valid URL.")

    return url


def get_domain(url):
    parsed = urlparse(url)
    return parsed.hostname.lower().rstrip(".") if parsed.hostname else ""


def is_trusted_domain(domain):
    if not domain:
        return False

    return any(
        domain == trusted or domain.endswith("." + trusted)
        for trusted in TRUSTED_DOMAINS
    )

# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.hostname or ""
    domain = domain.lower()

    extracted = tldextract.extract(domain)

    tld = extracted.suffix.lower() if extracted.suffix else "unknown"

    try:
        ipaddress.ip_address(domain)
        has_ip = 1
    except ValueError:
        has_ip = 0

    suspicious_count = sum(
        1 for word in SUSPICIOUS_WORDS
        if word in url.lower()
    )

    is_shortened = int(domain in SHORTENED_DOMAINS)

    path_parts = [
        part for part in parsed.path.split("/")
        if part
    ]

    query_params = (
        len(parsed.query.split("&"))
        if parsed.query else 0
    )

    shortened_code_length = (
        len(path_parts[0])
        if is_shortened and path_parts
        else 0
    )

    features = {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(
            not c.isalnum() for c in url
        ),
        "num_subdirs": len(path_parts),
        "num_query_params": query_params,
        "has_at_symbol": int("@" in url),
        "has_ip_address": has_ip,
        "num_suspicious_words": suspicious_count,
        "domain_length": len(domain),
        "tld": tld,
        "uses_https": int(parsed.scheme.lower() == "https"),
        "domain_age_days": -1,
        "domain_reputation_score": -1,
        "is_shortened": is_shortened,
        "shortened_code_length": shortened_code_length,
    }

    return features

# --------------------------------------------------
# PREPARE MODEL INPUT
# --------------------------------------------------

def prepare_model_input(features):
    df = pd.DataFrame([features])

    # Handle TLD values
    df["tld"] = df["tld"].apply(
        lambda value: value if value in top_tlds else "other"
    )

    # One-hot encoding
    df = pd.get_dummies(df, columns=["tld"])

    # Fill unknown domain information
    for column in ["domain_age_days", "domain_reputation_score"]:
        if column in df.columns:
            median_value = medians.get(column, 0)
            df[column] = df[column].replace(-1, median_value)

    # Match the exact training feature order
    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return df

# --------------------------------------------------
# EXPLANATIONS
# --------------------------------------------------

def generate_explanations(features):
    signals = []
    info = []

    if features["is_shortened"]:
        signals.append(
            "URL uses a shortened link service."
        )

    if features["has_ip_address"]:
        signals.append(
            "URL uses an IP address instead of a domain name."
        )

    if features["has_at_symbol"]:
        signals.append(
            "URL contains an @ symbol, which may hide the destination."
        )

    if not features["uses_https"]:
        signals.append(
            "URL does not use HTTPS encryption."
        )

    if features["num_suspicious_words"] > 0:
        signals.append(
            "URL contains suspicious keywords."
        )

    if features["num_hyphens"] >= 3:
        signals.append(
            "URL contains multiple hyphens."
        )

    info.append(
        "Domain age could not be verified during live scanning."
    )

    info.append(
        "Domain reputation information is unavailable for this scan."
    )

    return signals, info

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_url(url):
    domain = get_domain(url)

    # Trusted-domain override
    if is_trusted_domain(domain):
        return {
            "prediction": "benign",
            "result": "SAFE - Trusted Domain",
            "confidence": 0.0,
            "signals": [],
            "info": [
                "This domain belongs to the configured trusted-domain list.",
                "The trusted-domain rule bypassed the machine-learning prediction.",
            ],
        }

    features = extract_features(url)
    model_input = prepare_model_input(features)

    prediction = model.predict(model_input)[0]

    probabilities = model.predict_proba(model_input)[0]
    classes = list(model.classes_)

    malicious_index = None

    for index, label in enumerate(classes):
        if str(label).lower() == "malicious":
            malicious_index = index
            break

    if malicious_index is None:
        for index, label in enumerate(classes):
            if str(label).lower() in ["1", "malicious"]:
                malicious_index = index
                break

    if malicious_index is None:
        malicious_index = len(classes) - 1

    malicious_score = float(
        probabilities[malicious_index] * 100
    )

    is_malicious = (
        str(prediction).lower() == "malicious"
        or prediction == 1
    )

    signals, info = generate_explanations(features)

    if is_malicious:
        result = "MALICIOUS / SUSPICIOUS URL"
    else:
        result = "SAFE URL"

    return {
        "prediction": "malicious" if is_malicious else "benign",
        "result": result,
        "confidence": round(malicious_score, 2),
        "signals": signals,
        "info": info,
    }

# --------------------------------------------------
# FLASK ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prediction = None
    confidence = 0
    signals = []
    info = []
    url = ""

    if request.method == "POST":
        submitted_url = request.form.get("url", "").strip()
        url = submitted_url

        try:
            normalized_url = normalize_url(submitted_url)
            url = normalized_url

            output = predict_url(normalized_url)

            result = output["result"]
            prediction = output["prediction"]
            confidence = output["confidence"]
            signals = output["signals"]
            info = output["info"]

        except Exception as error:
            logging.exception("URL scanning error")

            result = "Unable to analyze URL"
            prediction = "error"
            confidence = 0
            signals = []
            info = [str(error)]

    return render_template(
        "index.html",
        result=result,
        prediction=prediction,
        confidence=confidence,
        signals=signals,
        info=info,
        url=url,
    )


if __name__ == "__main__":
    app.run(debug=True)
