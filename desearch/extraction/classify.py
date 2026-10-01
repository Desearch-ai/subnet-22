import re
from urllib.parse import unquote, urlsplit

from lxml import etree

from desearch.extraction.structured import Structured, is_article_type

LONG_PARAGRAPH = 120

LOCALES = {
    "en",
    "fr",
    "de",
    "es",
    "it",
    "nl",
    "pt",
    "pl",
    "ru",
    "ja",
    "zh",
    "ko",
    "sv",
    "da",
    "no",
    "nb",
    "fi",
    "tr",
    "ar",
    "he",
    "cs",
    "hu",
    "ro",
    "el",
    "uk",
    "id",
    "vi",
    "th",
}
INDEX_FILES = {"index", "index.html", "index.htm", "index.php", "home", "default.aspx"}
FILE_SUFFIX = re.compile(r"\.(html?|php|aspx?|jsp|cfm)$")
REGIONAL_LOCALE = re.compile(r"^[a-z]{2}[-_][a-z]{2}$")

LISTING_SEGMENTS = {
    "category",
    "categories",
    "tag",
    "tags",
    "archive",
    "archives",
    "topic",
    "topics",
    "author",
    "authors",
    "product-category",
    "product-tag",
    "collections",
    "label",
    "search",
}
SECTION_SEGMENTS = {
    "blog",
    "blogs",
    "news",
    "articles",
    "stories",
    "insights",
    "posts",
    "press",
    "press-releases",
    "newsroom",
    "events",
    "resources",
    "publications",
    "podcasts",
    "videos",
    "library",
    "journal",
    "magazine",
    "guides",
}
CONTENT_PARENTS = SECTION_SEGMENTS | {
    "article",
    "story",
    "post",
    "opinion",
    "editorial",
    "learn",
    "podcast",
    "updates",
}
COMPANY_SEGMENTS = {
    "about",
    "about-us",
    "aboutus",
    "who-we-are",
    "our-story",
    "our-team",
    "team",
    "leadership",
    "management",
    "board",
    "board-of-directors",
    "people",
    "our-people",
    "staff",
    "careers",
    "career",
    "jobs",
    "join-us",
    "work-with-us",
    "contact",
    "contact-us",
    "contactus",
    "company",
    "history",
    "mission",
    "our-mission",
    "investors",
    "investor-relations",
    "locations",
    "offices",
}
LANDING_SEGMENTS = {
    "pricing",
    "plans",
    "features",
    "product",
    "products",
    "solutions",
    "solution",
    "services",
    "service",
    "platform",
    "shop",
    "store",
    "buy",
    "demo",
    "signup",
    "sign-up",
    "get-started",
    "enterprise",
    "integrations",
    "use-cases",
    "ecommerce",
    "download",
    "downloads",
    "app",
    "offers",
    "deals",
    "free-trial",
    "industries",
}
BOILERPLATE_PAGES = {
    "privacy",
    "privacy-policy",
    "terms",
    "terms-of-use",
    "terms-of-service",
    "terms-and-conditions",
    "tos",
    "legal",
    "cookies",
    "cookie-policy",
    "cookies-policy",
    "disclaimer",
    "dmca",
    "dmca-disclaimer",
    "sitemap",
    "login",
    "signin",
    "sign-in",
    "register",
    "cart",
    "checkout",
    "account",
    "my-account",
    "404",
}
LISTING_TYPES = {"CollectionPage", "SearchResultsPage"}
COMPANY_TYPES = {"AboutPage", "ContactPage"}
LANDING_TYPES = {
    "Product",
    "ProductGroup",
    "IndividualProduct",
    "Service",
    "SoftwareApplication",
    "MobileApplication",
    "WebApplication",
}

CONTENT_PARAGRAPHS = etree.XPath(
    "//p[not(ancestor::nav or ancestor::header or ancestor::footer or ancestor::aside)]"
)
ARTICLE_ELEMENTS = etree.XPath("count(//article)")


def path_segments(url: str) -> list[str]:
    try:
        parts = urlsplit(url or "")
    except ValueError:
        return []
    segments = [s for s in unquote(parts.path).lower().split("/") if s.strip()]
    if segments and (segments[0] in LOCALES or REGIONAL_LOCALE.match(segments[0])):
        segments = segments[1:]
    return [
        FILE_SUFFIX.sub("", s) if i == len(segments) - 1 else s
        for i, s in enumerate(segments)
    ]


def words(segment: str) -> list[str]:
    return [w for w in re.split(r"[-_+.\s]+", segment) if w]


def is_home(segments: list[str]) -> bool:
    if not segments:
        return True
    return len(segments) == 1 and segments[0] in INDEX_FILES


def is_paginated(url: str, segments: list[str]) -> bool:
    for i, segment in enumerate(segments[:-1]):
        if segment == "page" and segments[i + 1].isdigit():
            return True
    query = urlsplit(url).query.lower()
    return bool(re.search(r"(^|&)(page|paged|pg)=\d+", query))


def paragraph_density(root: etree._Element) -> tuple[int, int]:
    count = chars = 0
    for paragraph in CONTENT_PARAGRAPHS(root):
        length = len(" ".join(paragraph.text_content().split()))
        if length >= LONG_PARAGRAPH:
            count += 1
            chars += length
    return count, chars


def classify(url: str, meta: Structured, root: etree._Element) -> str:
    segments = path_segments(url)
    if is_home(segments):
        return "home"
    if segments[-1] in BOILERPLATE_PAGES:
        return "other"

    types = set(meta.json_ld_types)
    article_typed = any(map(is_article_type, types))
    listing_typed = bool(types & LISTING_TYPES)
    long_count, long_chars = paragraph_density(root)
    dense = long_count >= 3 and long_chars >= 1200
    strong_article = article_typed and (
        long_chars >= 1500
        if listing_typed
        else meta.article_published or long_chars >= 600
    )

    listing_path = (
        any(s in LISTING_SEGMENTS for s in segments)
        or segments[-1] in SECTION_SEGMENTS
        or is_paginated(url, segments)
    )
    if listing_path and not (strong_article and meta.article_published):
        return "listing"
    if strong_article or (meta.og_type == "article" and meta.article_published):
        return "article"
    if types & COMPANY_TYPES or any(
        s in COMPANY_SEGMENTS or s.startswith("about-") for s in segments
    ):
        return "company"
    if types & LANDING_TYPES or any(s in LANDING_SEGMENTS for s in segments):
        return "landing"
    in_content_section = any(s in CONTENT_PARENTS for s in segments[:-1])
    slug_like = sum(w.isalpha() for w in words(segments[-1])) >= 4
    if dense and (meta.article_published or in_content_section or slug_like):
        return "article"
    if listing_typed:
        return "listing"
    if len(segments) == 1 and len(words(segments[0])) <= 3:
        return "landing"
    if ARTICLE_ELEMENTS(root) >= 6:
        return "listing"
    return "other"
