"""PhishGuard - URL Feature Extractor Module.

This module analyzes a URL string and computes only the features that can be
legitimately and reliably calculated from the URL string alone (lexical features).

Features requiring webpage content (HTML DOM), active network scans, DNS queries,
SSL certificate inspection, or third-party APIs (Alexa, PageRank, PhishTank)
are deliberately NOT calculated or fabricated, to prevent false/misleading predictions.

Reference:
    Mohammad, R., Thabtah, F., & McCluskey, L. (2012).
    "An assessment of features related to phishing websites using an automated technique"
"""

import ipaddress
import re
from urllib.parse import urlparse

# Known URL shortening services documented in phishing research
COMMON_SHORTENERS = {
    "bit.ly", "goo.gl", "shorte.st", "go2l.ink", "x.co", "ow.ly", "t.co",
    "tinyurl", "tinyurl.com", "tr.im", "is.gd", "cli.gs", "yfrog.com",
    "migre.me", "ff.im", "tiny.cc", "url4.eu", "twit.ac", "su.pr",
    "twurl.nl", "snipurl.com", "short.to", "budurl.com", "ping.fm",
    "post.ly", "just.as", "bkite.com", "snipr.com", "fic.kr", "loopt.us",
    "doiop.com", "short.ie", "kl.am", "wp.me", "rubyurl.com", "om.ly",
    "to.ly", "bit.do", "t.ly", "lnkd.in", "db.tt", "qr.ae", "adf.ly",
    "bitly.com", "cur.lv", "ity.im", "q.gs", "linkrex.net"
}

# The 22 features from the UCI dataset that CANNOT be derived from the URL alone
# grouped by the external dependency required
EXTERNAL_DEPENDENCY_FEATURES = {
    "network_and_dns": [
        "sslfinal_state",             # Requires SSL handshake and certificate authority audit
        "domain_registration_length", # Requires WHOIS query for domain expiration date
        "port",                       # Requires active port scan for open non-standard ports
        "age_of_domain",              # Requires WHOIS query for domain creation date
        "dnsrecord",                  # Requires live DNS resolution lookup
        "redirect"                    # Requires issuing HTTP requests to count redirect hops
    ],
    "webpage_content_dom": [
        "favicon",                    # Requires downloading HTML to inspect <link rel='icon'>
        "request_url",                # Requires HTML parsing for external image/media origins
        "url_of_anchor",              # Requires HTML parsing for <a> tag href destinations
        "links_in_tags",              # Requires HTML parsing for <meta>, <script>, <link>
        "sfh",                        # Requires HTML parsing for Server Form Handler action
        "submitting_to_email",        # Requires HTML/JS parsing for mail() or mailto:
        "abnormal_url",               # Requires comparing WHOIS record with webpage host
        "on_mouseover",               # Requires JavaScript AST parsing for onMouseOver tampering
        "rightclick",                 # Requires JavaScript parsing for contextmenu blocking
        "popupwindow",                # Requires JavaScript parsing for pop-up dialogs
        "iframe"                      # Requires HTML parsing for invisible <iframe> frames
    ],
    "external_third_party_apis": [
        "web_traffic",                # Requires Alexa / SimilarWeb rank API (< 100,000)
        "page_rank",                  # Requires Google PageRank lookup API
        "google_index",               # Requires Google Search index API
        "links_pointing_to_page",     # Requires backlink index (Moz / Ahrefs / Alexa)
        "statistical_report"          # Requires querying PhishTank / StopBadware feeds
    ]
}

ALL_MISSING_FEATURES = (
    EXTERNAL_DEPENDENCY_FEATURES["network_and_dns"] +
    EXTERNAL_DEPENDENCY_FEATURES["webpage_content_dom"] +
    EXTERNAL_DEPENDENCY_FEATURES["external_third_party_apis"]
)


def validate_url(url_string):
    """Validate that the input is a well-formed URL string.

    Returns:
        tuple (bool, str, object): (is_valid, error_message, parsed_url)
    """
    if not isinstance(url_string, str):
        return False, "Input must be a string", None

    cleaned_url = url_string.strip()
    if not cleaned_url:
        return False, "URL cannot be empty or whitespace only", None

    # Prepend scheme if missing so urlparse works consistently
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", cleaned_url):
        cleaned_url = "http://" + cleaned_url

    try:
        parsed = urlparse(cleaned_url)
        if not parsed.netloc:
            return False, "URL must contain a valid domain or IP address", None
        return True, "", parsed
    except Exception as exc:
        return False, f"Malformed URL: {exc}", None


def extract_lexical_features(url_string):
    """Extract lexical URL features strictly from the URL string.

    Returns:
        dict: Mapping of feature names to values {-1, 0, 1}.
    """
    is_valid, err, parsed = validate_url(url_string)
    if not is_valid:
        raise ValueError(err)

    raw_url = url_string.strip()
    host = parsed.netloc.lower()
    # Strip port if present for host checks
    host_without_port = host.split(":")[0]

    features = {}

    # 1. having_ip_address: If an IP address is used in domain -> Phishing (-1), else Legitimate (1)
    is_ip = False
    try:
        ipaddress.ip_address(host_without_port)
        is_ip = True
    except ValueError:
        # Check hexadecimal or octal IP format (e.g. 0x7f.0.0.1)
        if re.match(r"^0x[0-9a-fA-F]+", host_without_port):
            is_ip = True
    features["having_ip_address"] = -1 if is_ip else 1

    # 2. url_length: < 54 -> 1 (Legitimate), 54-75 -> 0 (Suspicious), > 75 -> -1 (Phishing)
    url_len = len(raw_url)
    if url_len < 54:
        features["url_length"] = 1
    elif 54 <= url_len <= 75:
        features["url_length"] = 0
    else:
        features["url_length"] = -1

    # 3. shortining_service: Uses known shortener -> Phishing (-1), else Legitimate (1)
    is_shortener = any(
        host_without_port == s or host_without_port.endswith("." + s)
        for s in COMMON_SHORTENERS
    )
    features["shortining_service"] = -1 if is_shortener else 1

    # 4. having_at_symbol: Presence of '@' symbol -> Phishing (-1), else Legitimate (1)
    features["having_at_symbol"] = -1 if "@" in raw_url else 1

    # 5. double_slash_redirecting: If '//' appears anywhere in path -> Phishing (-1), else Legitimate (1)
    # The normal '//' is immediately after http: or https:
    scheme_end = raw_url.find("//")
    if scheme_end != -1:
        path_after_scheme = raw_url[scheme_end + 2:]
        has_internal_double_slash = "//" in path_after_scheme
    else:
        has_internal_double_slash = False
    features["double_slash_redirecting"] = -1 if has_internal_double_slash else 1

    # 6. prefix_suffix: If '-' (dash) appears in domain name -> Phishing (-1), else Legitimate (1)
    features["prefix_suffix"] = -1 if "-" in host_without_port else 1

    # 7. having_sub_domain: Count dots in hostname (excluding 'www')
    # 1 dot -> Legitimate (1), 2 dots -> Suspicious (0), >= 3 dots -> Phishing (-1)
    cleaned_host = host_without_port
    if cleaned_host.startswith("www."):
        cleaned_host = cleaned_host[4:]
    dot_count = cleaned_host.count(".")
    if dot_count <= 1:
        features["having_sub_domain"] = 1
    elif dot_count == 2:
        features["having_sub_domain"] = 0
    else:
        features["having_sub_domain"] = -1

    # 8. https_token: If 'https' is part of the domain name -> Phishing (-1), else Legitimate (1)
    # e.g., 'http://https-paypal-login.com' uses 'https' in the domain to deceive users
    features["https_token"] = -1 if "https" in host_without_port else 1

    return features


def inspect_feature_coverage(model_expected_features, extracted_features):
    """Compare extracted features against the model's required 30 features.

    Returns:
        dict: Detailed breakdown of available vs missing features.
    """
    extracted_set = set(extracted_features.keys())
    expected_set = set(model_expected_features)

    available = {f: extracted_features[f] for f in model_expected_features if f in extracted_set}
    missing = [f for f in model_expected_features if f not in extracted_set]

    coverage_pct = (len(available) / len(model_expected_features)) * 100

    return {
        "is_complete": len(missing) == 0,
        "total_required": len(model_expected_features),
        "total_available": len(available),
        "total_missing": len(missing),
        "coverage_percentage": round(coverage_pct, 2),
        "available_features": available,
        "missing_features": missing,
        "missing_by_category": EXTERNAL_DEPENDENCY_FEATURES
    }
