
import re
import ipaddress
import html
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import joblib
import tldextract


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="URL Shield | Intelligent URL Security",
    page_icon="🛡️",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model_files():
    model = joblib.load(BASE_DIR / "model_rf.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    medians = joblib.load(BASE_DIR / "medians.pkl")
    top_tlds = joblib.load(BASE_DIR / "top_tlds.pkl")

    return model, feature_columns, medians, top_tlds


try:
    model, feature_columns, medians, top_tlds = load_model_files()
    model_loaded = True
except Exception as error:
    model_loaded = False
    model_error = str(error)


# ==================================================
# CONSTANTS
# These match your original training script
# ==================================================

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

SUSPICIOUS_WORDS = [
    "login", "verify", "account", "update", "secure",
    "banking", "confirm", "signin", "webscr",
    "password", "free", "gift", "prize", "win",
    "lottery", "claim", "urgent", "suspend"
]

SHORTENED_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl",
    "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
    "cutt.ly", "shorte.st", "tiny.cc", "bl.ink",
    "soo.gd", "s2r.co", "clck.ru", "v.gd"
}


# ==================================================
# URL VALIDATION
# ==================================================

def normalize_url(url):
    url = url.strip()

    if not url:
        raise ValueError("Please enter a URL.")

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)

    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    if not parsed.hostname:
        raise ValueError("Please enter a valid URL.")

    try:
        parsed.port
    except ValueError:
        raise ValueError("Invalid port number in URL.")

    return url


def get_domain(url):
    parsed = urlparse(url)
    return parsed.hostname.lower().rstrip(".") if parsed.hostname else ""


def is_trusted_domain(domain):
    return any(
        domain == trusted or domain.endswith("." + trusted)
        for trusted in TRUSTED_DOMAINS
    )


# ==================================================
# FEATURE EXTRACTION
# Matches extract_features.py used during training
# ==================================================

def extract_url_features(url):
    parsed = urlparse(url)

    domain_full = parsed.netloc.lower()

    extracted = tldextract.extract(url)

    registered_domain = (
        f"{extracted.domain}.{extracted.suffix}"
        if extracted.suffix
        else extracted.domain
    )

    suspicious_count = sum(
        1 for word in SUSPICIOUS_WORDS
        if word in url.lower()
    )

    is_shortened = int(
        registered_domain in SHORTENED_DOMAINS
    )

    try:
        ipaddress.ip_address(parsed.hostname)
        has_ip = 1
    except (ValueError, TypeError):
        has_ip = 0

    features = {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": len(
            re.findall(r"[^a-zA-Z0-9]", url)
        ),
        "num_subdirs": parsed.path.count("/"),
        "num_query_params": (
            len(parsed.query.split("&"))
            if parsed.query else 0
        ),
        "has_at_symbol": int("@" in url),
        "has_ip_address": has_ip,
        "num_suspicious_words": suspicious_count,
        "domain_length": len(domain_full),
        "tld": extracted.suffix,
        "uses_https": int(parsed.scheme.lower() == "https"),
        "domain_age_days": -1,
        "domain_reputation_score": -1,
        "is_shortened": is_shortened,
        "shortened_code_length": (
            len(parsed.path.strip("/"))
            if is_shortened else 0
        )
    }

    return features


# ==================================================
# PREPARE MODEL INPUT
# ==================================================

def prepare_model_input(features):
    df = pd.DataFrame([features])

    df["tld"] = df["tld"].where(
        df["tld"].isin(top_tlds),
        "other"
    )

    df = pd.get_dummies(
        df,
        columns=["tld"],
        prefix="tld"
    )

    for column in [
        "domain_age_days",
        "domain_reputation_score"
    ]:
        if column in df.columns:
            median_value = medians.get(column, 0)
            df[column] = df[column].replace(
                -1, median_value
            )

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return df


# ==================================================
# EXPLANATIONS
# ==================================================

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


# ==================================================
# PREDICTION
# ==================================================

def predict_url(url):

    domain = get_domain(url)

    if is_trusted_domain(domain):
        return {
            "prediction": "benign",
            "result": "TRUSTED DOMAIN - ALLOWLIST MATCH",
            "confidence": 0.0,
            "signals": [],
            "info": [
                "This domain belongs to the configured trusted-domain list.",
                "The trusted-domain rule bypassed the machine-learning prediction.",
                "This is not a guarantee that every URL on the domain is safe."
            ]
        }

    features = extract_url_features(url)

    model_input = prepare_model_input(features)

    prediction = model.predict(model_input)[0]

    probabilities = model.predict_proba(model_input)[0]

    classes = list(model.classes_)

    malicious_index = next(
        (
            i for i, label in enumerate(classes)
            if str(label).lower() in ("1", "malicious")
        ),
        None
    )

    if malicious_index is None:
        raise ValueError(
            "Could not identify the malicious class in the trained model."
        )

    malicious_score = float(
        probabilities[malicious_index] * 100
    )

    is_malicious = (
        str(prediction).lower() in ("1", "malicious")
    )

    signals, info = generate_explanations(features)

    return {
        "prediction": "malicious" if is_malicious else "benign",
        "result": (
            "MALICIOUS / SUSPICIOUS URL"
            if is_malicious else "MODEL CLASSIFIED URL AS BENIGN"
        ),
        "confidence": round(malicious_score, 2),
        "signals": signals,
        "info": info
    }


# ==================================================
# CUSTOM CSS - URL SHIELD DESIGN
# ==================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            ellipse at 50% -20%,
            rgba(49, 100, 220, 0.18),
            transparent 55%
        ),
        #080c17;
    color: #edf3ff;
}

.block-container {
    max-width: 950px;
    padding-top: 2rem;
}

h1, h2, h3, p, label {
    color: #edf3ff;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #edf3ff;
}

.brand span {
    color: #80b5ff;
}

.status {
    color: #20c997;
    text-align: right;
    font-size: 13px;
    padding-top: 10px;
}

.hero {
    text-align: center;
    padding: 45px 0 35px 0;
}

.hero-badge {
    display: inline-block;
    color: #90baff;
    background: rgba(59,130,246,0.09);
    border: 1px solid #28446e;
    padding: 8px 15px;
    border-radius: 30px;
    font-size: 11px;
    letter-spacing: 1px;
}

.hero-title {
    font-size: clamp(32px, 5vw, 50px);
    font-weight: 800;
    line-height: 1.2;
    margin-top: 22px;
}

.hero-title span {
    color: #80aaff;
}

.hero-description {
    color: #8493ab;
    font-size: 15px;
    line-height: 1.8;
    max-width: 650px;
    margin: 15px auto;
}

.scanner-card {
    background: linear-gradient(180deg, #111b2d, #0d1524);
    border: 1px solid #233149;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
}

.stTextInput input {
    background: #080e1a !important;
    color: white !important;
    border: 1px solid #2a3951 !important;
    border-radius: 10px !important;
}

.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(135deg, #3478f6, #5b65ed);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    width: 100%;
    min-height: 45px;
}

.result-card {
    background: #0a1220;
    border: 1px solid #233149;
    border-radius: 15px;
    padding: 25px;
    margin-top: 25px;
}

.safe {
    border-color: rgba(32,201,151,0.5);
}

.danger {
    border-color: rgba(255,100,124,0.5);
}

.result-title {
    font-size: 21px;
    font-weight: 800;
}

.safe-text {
    color: #20c997;
}

.danger-text {
    color: #ff647c;
}

.muted {
    color: #8493ab;
    font-size: 13px;
}

.footer {
    text-align: center;
    color: #8493ab;
    font-size: 12px;
    margin-top: 50px;
    padding: 25px 0;
    border-top: 1px solid #233149;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# HEADER
# ==================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        '<div class="brand">🛡 URL <span>SHIELD</span></div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div class="status">● ML SCANNER</div>',
        unsafe_allow_html=True
    )


# ==================================================
# HERO SECTION
# ==================================================

st.markdown("""
<div class="hero">

    <div class="hero-badge">
        ✦ INTELLIGENT URL SECURITY
    </div>

    <div class="hero-title">
        Secure Every Click.<br>
        <span>Detect Digital Threats.</span>
    </div>

    <div class="hero-description">
        Analyze URLs with machine learning to identify
        potentially malicious links, suspicious patterns,
        and security risks before interacting with them.
    </div>

</div>
""", unsafe_allow_html=True)


# ==================================================
# CHECK MODEL
# ==================================================

if not model_loaded:
    st.error("Unable to load the trained ML model.")
    st.code(model_error)
    st.stop()


# ==================================================
# URL SCANNER
# ==================================================

st.markdown("""
<div class="scanner-card">
    <h3>⌕ URL Analysis</h3>
    <p class="muted">
        Enter a URL to perform a security scan.
    </p>
</div>
""", unsafe_allow_html=True)

with st.form("url_scanner"):

    submitted_url = st.text_input(
        "Enter website URL",
        placeholder="https://example.com"
    )

    scan_button = st.form_submit_button(
        "Analyze URL →",
        use_container_width=True
    )


# ==================================================
# DISPLAY RESULTS
# ==================================================

if scan_button:

    try:
        normalized_url = normalize_url(submitted_url)

        with st.spinner("Analyzing URL with machine learning..."):
            output = predict_url(normalized_url)

        prediction = output["prediction"]
        confidence = output["confidence"]

        safe = prediction == "benign"

        result_class = "safe" if safe else "danger"
        result_color = "safe-text" if safe else "danger-text"

        result_url = html.escape(normalized_url)

        st.markdown(
            f"""
            <div class="result-card {result_class}">

                <div class="result-title {result_color}">
                    {"✓" if safe else "⚠"}
                    {html.escape(output["result"])}
                </div>

                <p class="muted">
                    {"The model classified this URL as benign."
                    if safe else
                    "The model classified this URL as malicious."}
                </p>

                <br>

                <p class="muted">ANALYZED URL</p>

                <p style="overflow-wrap:anywhere;color:#b5c4da;">
                    {result_url}
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Malicious prediction score")

        st.progress(
            min(max(confidence, 0.0), 100.0) / 100.0
        )

        st.markdown(
            f"**{confidence:.2f}%**"
        )

        left, right = st.columns(2)

        with left:
            st.markdown("### ⚠ Detection Signals")

            if output["signals"]:
                for signal in output["signals"]:
                    st.warning(signal)
            else:
                st.success(
                    "No obvious suspicious URL patterns were detected."
                )

        with right:
            st.markdown("### ⓘ Additional Information")

            for item in output["info"]:
                st.info(item)

        st.caption(
            "Note: A benign classification does not guarantee "
            "that a website is safe. Do not open suspicious links."
        )

    except Exception as error:
        st.error("Unable to analyze URL.")
        st.exception(error)


# ==================================================
# FOOTER
# ==================================================

st.markdown("""
<div class="footer">

    <strong>URL SHIELD</strong><br>

    Intelligent Detection of Malicious Shortened URLs<br>

    Machine Learning Security Research Project<br><br>

    Predictions are not a guarantee of website safety.
    Always verify suspicious links before opening them.

</div>
""", unsafe_allow_html=True)
