import re

REAL_CONTENT_CHARS = 3000
MARKER_SCAN_CHARS = 200_000
TITLE_PREFIX_TEXT_CHARS = 500

TITLE_OPEN = re.compile(r"<title", re.I)
TITLE_CLOSE = re.compile(r"</title>", re.I)
CHALLENGE_PHRASE = (
    r"\s*(?:just a moment|one moment|attention required|access denied|access to this page has been"
    r" denied|security verification|security check|human verification|bot verification"
    r"|are you a (?:human|robot)|verify(?:ing)? (?:that )?you are (?:a )?human|robot check"
    r"|pardon our interruption|checking your browser|one more step|request (?:blocked"
    r"|rejected|unsuccessful)|you have been blocked|ddos-guard|vercel security"
    r" checkpoint|captcha|403 forbidden|forbidden|error 1020)"
)
CHALLENGE_TITLE = re.compile(
    CHALLENGE_PHRASE
    + r"[\s.!?\u2026]*(?:$|(?P<suffix>(?:\||\s[-\u2013\u2014\u00b7]\s).*))",
    re.I,
)
VENDOR_NAME = re.compile(
    r"\b(?:cloudflare|sucuri|incapsula|imperva|ddos-guard|vercel|akamai|perimeterx"
    r"|human security|datadome|kasada)\b",
    re.I,
)
CHALLENGE_PREFIX = re.compile(CHALLENGE_PHRASE + r"\b", re.I)
VENDOR_MARKERS = (
    "cf-browser-verification",
    "cf_chl_opt",
    "cf-error-details",
    "/cdn-cgi/challenge-platform/h/",
    "_incapsula_resource",
    "incapsula incident id",
    "px-captcha",
    "captcha-delivery.com",
    "sec-if-cpt",
    "errors.edgesuite.net",
    "awswaf",
    "aws-waf-token",
    "sucuri website firewall",
    "ddos-guard",
    "kpsdk",
    "vercel security checkpoint",
)
CHALLENGE_PHRASES = (
    "verify you are human",
    "verify you are a human",
    "verifies you are not a bot",
    "are you a robot",
    "not a robot",
    "enable javascript and cookies to continue",
    "checking your browser",
    "checking if the site connection is secure",
    "request is being verified",
    "unusual traffic",
    "you have been blocked",
    "security service to protect",
    "complete the security check",
    "request unsuccessful",
    "press & hold",
    "press and hold",
    "access to this page has been denied",
)
WEAK_PHRASES = ("captcha", "access denied", "forbidden", "too many requests")
SCRIPT_ONLY_PHRASES = (
    "enable javascript",
    "javascript is disabled",
    "javascript is required",
    "requires javascript",
    "turn on javascript",
)
REFUSAL_STATUSES = frozenset({401, 403, 407, 429})


def page_title(head: str) -> str:
    opened = TITLE_OPEN.search(head)
    start = head.find(">", opened.end()) if opened else -1
    closed = TITLE_CLOSE.search(head, start + 1) if start >= 0 else None
    return " ".join(head[start + 1 : closed.start()].split()) if closed else ""


def looks_blocked(status: int, html: str, text: str, title: str | None = None) -> bool:
    visible = " ".join((text or "").split()).lower()
    if len(visible) > REAL_CONTENT_CHARS:
        return False

    head = (html or "")[:MARKER_SCAN_CHARS]
    refused = status in REFUSAL_STATUSES or status == 503
    title = page_title(head) if title is None else " ".join(title.split())
    challenge = CHALLENGE_TITLE.match(title)
    if challenge and (
        not challenge["suffix"] or VENDOR_NAME.search(challenge["suffix"])
    ):
        return True
    if CHALLENGE_PREFIX.match(title) and (
        refused or len(visible) < TITLE_PREFIX_TEXT_CHARS
    ):
        return True

    lowered = head.lower()
    if any(marker in lowered for marker in VENDOR_MARKERS) and (
        refused or len(visible) < 200
    ):
        return True
    if any(phrase in visible for phrase in CHALLENGE_PHRASES) and len(visible) < 1000:
        return True
    if any(phrase in visible for phrase in SCRIPT_ONLY_PHRASES) and len(visible) < 200:
        return True
    if any(phrase in visible for phrase in WEAK_PHRASES) and len(visible) < 200:
        return refused or len(visible) < 80
    return status in REFUSAL_STATUSES and len(visible) < 200
