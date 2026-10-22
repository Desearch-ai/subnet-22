"""JSON-LD: the page's types, authors, breadcrumbs, entity, the condensed ld field and its paywall flag."""

import json
import math
import re

from lxml import etree

from desearch.extraction.structured.dates import read_date
from desearch.extraction.structured.html import MAX_FIELD_CHARS, absolute_url, squash

MAX_NAME_CHARS = 300
MAX_LABEL_CHARS = 100
MAX_BREADCRUMBS = 10
MAX_SAME_AS = 10
MAX_ENTITY_CHARS = 4096
MAX_LD_CHARS = 16384
MAX_LD_ITEMS = 50
MAX_RECIPE_ITEMS = 100
MAX_LD_FALLBACK_CHARS = 256_000
TAGS = re.compile(r"<[^>]+>")
LD_WRAPPERS = re.compile(r"<!--|-->|//\s*<!\[CDATA\[|//\s*\]\]>|<!\[CDATA\[|\]\]>")
TRAILING_COMMA = re.compile(r",\s*([}\]])")
LD_TYPE = re.compile(r'"@type"\s{0,20}:\s{0,20}(\[[^\[\]{}]{0,500}\]|"[^"]{0,200}")')
LD_TYPE_NAME = re.compile(r'"([^"]{1,200})"')
LD_SCRIPTS = etree.XPath(
    "//script[contains(translate(@type, 'LDJSON', 'ldjson'), 'ld+json')]"
)
ENTITY_TYPES = frozenset(
    (
        "Organization",
        "Corporation",
        "NGO",
        "Consortium",
        "Cooperative",
        "Project",
        "ResearchOrganization",
        "NewsMediaOrganization",
        "EducationalOrganization",
        "CollegeOrUniversity",
        "HighSchool",
        "School",
        "GovernmentOrganization",
        "PoliticalParty",
        "SportsOrganization",
        "SportsTeam",
        "MedicalOrganization",
        "Hospital",
        "Library",
        "Museum",
        "PerformingGroup",
        "MusicGroup",
        "Airline",
        "LocalBusiness",
        "OnlineBusiness",
        "OnlineStore",
        "Store",
        "Restaurant",
        "FoodEstablishment",
        "Hotel",
        "LodgingBusiness",
        "ProfessionalService",
        "LegalService",
        "FinancialService",
        "RealEstateAgent",
        "AutomotiveBusiness",
        "HealthAndBeautyBusiness",
        "HomeAndConstructionBusiness",
        "EntertainmentBusiness",
        "Dentist",
        "Physician",
        "Person",
    )
)
PRODUCT_TYPES = frozenset(
    ("Product", "ProductGroup", "IndividualProduct", "ProductModel")
)
GTIN_KEYS = ("gtin", "gtin13", "gtin12", "gtin14", "gtin8")


def is_article_type(ld_type: str) -> bool:
    return ld_type.endswith("Article") or ld_type in {
        "BlogPosting",
        "LiveBlogPosting",
        "DiscussionForumPosting",
        "Report",
        "Recipe",
        "HowTo",
    }


def load_json_ld(raw: str) -> object:
    text = LD_WRAPPERS.sub("", raw).strip()
    for candidate in (text, TRAILING_COMMA.sub(r"\1", text)):
        try:
            return json.loads(candidate, strict=False)
        except (ValueError, RecursionError):
            continue
    return None


def flatten_ld(data: object, depth: int = 0) -> list[dict]:
    if depth > 4:
        return []
    if isinstance(data, list):
        return [node for item in data for node in flatten_ld(item, depth + 1)]
    if not isinstance(data, dict):
        return []
    return [data, *flatten_ld(data.get("@graph"), depth + 1)]


def clean_type(value: str) -> str:
    return value.strip().rsplit("/", 1)[-1].rsplit(":", 1)[-1].strip()


def node_types(node: dict) -> list[str]:
    raw = node.get("@type")
    values = raw if isinstance(raw, list) else [raw]
    return [clean_type(v) for v in values if isinstance(v, str) and clean_type(v)]


def read_json_ld(root: etree._Element) -> tuple[list[dict], list[str]]:
    nodes: list[dict] = []
    types: list[str] = []
    for script in LD_SCRIPTS(root):
        raw = script.text or ""
        data = load_json_ld(raw)
        if data is None:
            if len(raw) <= MAX_LD_FALLBACK_CHARS:
                for found in LD_TYPE.findall(raw):
                    types.extend(clean_type(t) for t in LD_TYPE_NAME.findall(found))
            continue
        for node in flatten_ld(data):
            nodes.append(node)
            types.extend(node_types(node))
    unique = list(dict.fromkeys(t for t in types if t))
    return nodes, unique[:50]


def first_ld_date(nodes: list[dict], key: str) -> str:
    for node in nodes:
        value = node.get(key)
        if isinstance(value, list):
            value = value[0] if value else ""
        date = read_date(value)
        if date:
            return date
    return ""


def person_names(value: object, by_id: dict[str, dict]) -> list[str]:
    if isinstance(value, str):
        return [value] if not value.startswith("http") else []
    if isinstance(value, list):
        return [name for item in value[:10] for name in person_names(item, by_id)]
    if isinstance(value, dict):
        name = value.get("name")
        if not name and isinstance(value.get("@id"), str):
            name = by_id.get(value["@id"], {}).get("name")
        return [name] if isinstance(name, str) and name.strip() else []
    return []


def plain(value: object, limit: int = MAX_FIELD_CHARS) -> str:
    """A JSON-LD value as one line of text: strings as they are, numbers in their shortest form, names of objects."""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return squash(repr(value), limit)
    if isinstance(value, str):
        return squash(TAGS.sub(" ", value), limit)
    if isinstance(value, dict):
        return plain(
            value.get("name") or value.get("text") or value.get("@value"), limit
        )
    if isinstance(value, list) and value:
        return plain(value[0], limit)
    return ""


def strings(value: object, limit: int, count: int) -> list[str]:
    values = value if isinstance(value, list) else [value]
    found = (plain(v, limit) for v in values)
    return list(dict.fromkeys(v for v in found if v))[:count]


def read_breadcrumbs(nodes: list[dict]) -> list[str]:
    for node in nodes:
        if "BreadcrumbList" not in node_types(node):
            continue
        items = node.get("itemListElement")
        items = items if isinstance(items, list) else []
        ranked = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            position = item.get("position")
            rank = (
                int(position)
                if (isinstance(position, (int, float)) and math.isfinite(position))
                or (isinstance(position, str) and position.isdecimal())
                else i
            )
            name = plain(item.get("name"), MAX_LABEL_CHARS) or plain(
                item.get("item"), MAX_LABEL_CHARS
            )
            if name:
                ranked.append((rank, i, name))
        return [name for _, _, name in sorted(ranked)][:MAX_BREADCRUMBS]
    return []


def is_entity_type(ld_type: str) -> bool:
    return ld_type in ENTITY_TYPES or ld_type.endswith(("Organization", "Business"))


def read_address(value: object) -> dict:
    value = value[0] if isinstance(value, list) and value else value
    if isinstance(value, str):
        return {"street": squash(value, MAX_NAME_CHARS)} if value.strip() else {}
    if not isinstance(value, dict):
        return {}
    found = {
        "street": plain(value.get("streetAddress"), MAX_NAME_CHARS),
        "locality": plain(value.get("addressLocality"), MAX_LABEL_CHARS),
        "region": plain(value.get("addressRegion"), MAX_LABEL_CHARS),
        "postal_code": plain(value.get("postalCode"), MAX_LABEL_CHARS),
        "country": plain(value.get("addressCountry"), MAX_LABEL_CHARS),
    }
    return {k: v for k, v in found.items() if v}


def read_geo(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    found = {
        "lat": plain(value.get("latitude"), 30),
        "lon": plain(value.get("longitude"), 30),
    }
    return found if all(found.values()) else {}


def compact(found: dict) -> dict:
    return {k: v for k, v in found.items() if v not in ("", [], {}, None)}


def as_json(value: dict, limit: int) -> str:
    if not value:
        return ""
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return encoded if len(encoded) <= limit else ""


def image_url(value: object) -> str:
    return absolute_url(value.get("url") if isinstance(value, dict) else value, "")


def urls(value: object) -> list[str]:
    found = (
        absolute_url(v, "") for v in (value if isinstance(value, list) else [value])
    )
    return list(dict.fromkeys(u for u in found if u))


def read_entity(nodes: list[dict]) -> str:
    for node in nodes:
        types = [t for t in node_types(node) if is_entity_type(t)]
        if not types:
            continue
        found = compact(
            {
                "type": types[0],
                "name": plain(node.get("name"), MAX_NAME_CHARS),
                "legal_name": plain(node.get("legalName"), MAX_NAME_CHARS),
                "description": plain(node.get("description")),
                "url": absolute_url(node.get("url"), ""),
                "logo": image_url(node.get("logo")),
                "same_as": urls(node.get("sameAs"))[:MAX_SAME_AS],
                "address": read_address(node.get("address")),
                "telephone": plain(node.get("telephone"), MAX_LABEL_CHARS),
                "founding_date": read_date(node.get("foundingDate")),
                "geo": read_geo(node.get("geo")),
                "job_title": plain(node.get("jobTitle"), MAX_NAME_CHARS)
                if types[0] == "Person"
                else "",
                "affiliation": plain(
                    node.get("affiliation") or node.get("worksFor"), MAX_NAME_CHARS
                )
                if types[0] == "Person"
                else "",
            }
        )
        encoded = as_json(found, MAX_ENTITY_CHARS) or as_json(
            {k: v for k, v in found.items() if k != "description"}, MAX_ENTITY_CHARS
        )
        if encoded:
            return encoded
    return ""


def steps(value: object, count: int) -> list[str]:
    found: list[str] = []
    items = value if isinstance(value, list) else [value]
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("itemListElement"), list):
            found += steps(item["itemListElement"], count)
        else:
            text = (
                plain(item.get("text") or item.get("name"))
                if isinstance(item, dict)
                else plain(item)
            )
            if text:
                found.append(text)
        if len(found) >= count:
            break
    return found[:count]


def first_node(nodes: list[dict], wanted) -> dict | None:
    return next((n for n in nodes if any(wanted(t) for t in node_types(n))), None)


def read_ld(nodes: list[dict], article_nodes: list[dict]) -> str:
    found: dict = {}
    faq = first_node(nodes, lambda t: t == "FAQPage")
    if faq:
        questions = faq.get("mainEntity")
        pairs = []
        for question in questions if isinstance(questions, list) else [questions]:
            if isinstance(question, dict):
                answer = question.get("acceptedAnswer")
                pair = {
                    "q": plain(question.get("name")),
                    "a": plain(
                        answer.get("text") if isinstance(answer, dict) else answer
                    ),
                }
                if pair["q"] and pair["a"]:
                    pairs.append(pair)
        found["faq"] = pairs[:MAX_LD_ITEMS]
    howto = first_node(nodes, lambda t: t == "HowTo")
    if howto:
        found["howto"] = steps(howto.get("step"), MAX_LD_ITEMS)
    recipe = first_node(nodes, lambda t: t == "Recipe")
    if recipe:
        found["recipe"] = compact(
            {
                "ingredients": strings(
                    recipe.get("recipeIngredient"), MAX_FIELD_CHARS, MAX_RECIPE_ITEMS
                ),
                "steps": steps(recipe.get("recipeInstructions"), MAX_RECIPE_ITEMS),
                "prep_time": plain(recipe.get("prepTime"), 30),
                "cook_time": plain(recipe.get("cookTime"), 30),
                "total_time": plain(recipe.get("totalTime"), 30),
                "yield": plain(recipe.get("recipeYield"), MAX_LABEL_CHARS),
            }
        )
    product = first_node(nodes, lambda t: t in PRODUCT_TYPES)
    if product:
        offers = product.get("offers")
        offer = (
            offers[0]
            if isinstance(offers, list) and offers
            else offers
            if isinstance(offers, dict)
            else {}
        )
        offer = offer if isinstance(offer, dict) else {}
        rating = (
            product.get("aggregateRating")
            if isinstance(product.get("aggregateRating"), dict)
            else {}
        )
        found["product"] = compact(
            {
                "name": plain(product.get("name"), MAX_NAME_CHARS),
                "brand": plain(product.get("brand"), MAX_NAME_CHARS),
                "sku": plain(product.get("sku"), MAX_LABEL_CHARS),
                "gtin": next(
                    (
                        g
                        for g in (
                            plain(product.get(k), MAX_LABEL_CHARS) for k in GTIN_KEYS
                        )
                        if g
                    ),
                    "",
                ),
                "price": plain(offer.get("price") or offer.get("lowPrice"), 30),
                "currency": plain(offer.get("priceCurrency"), 10),
                "availability": plain(
                    offer.get("availability"), MAX_LABEL_CHARS
                ).rsplit("/", 1)[-1],
                "rating": plain(rating.get("ratingValue"), 30),
                "review_count": plain(
                    rating.get("reviewCount") or rating.get("ratingCount"), 30
                ),
            }
        )
    event = first_node(nodes, lambda t: t.endswith("Event"))
    if event:
        found["event"] = compact(
            {
                "name": plain(event.get("name"), MAX_NAME_CHARS),
                "start": read_date(event.get("startDate")),
                "end": read_date(event.get("endDate")),
                "place": plain(event.get("location"), MAX_NAME_CHARS),
            }
        )
    video = first_node(nodes, lambda t: t == "VideoObject")
    if video:
        found["video"] = compact(
            {
                "name": plain(video.get("name"), MAX_NAME_CHARS),
                "uploaded": read_date(video.get("uploadDate")),
                "duration": plain(video.get("duration"), 30),
            }
        )
    words = next(
        (
            plain(n.get("wordCount"), 20)
            for n in article_nodes
            if plain(n.get("wordCount"), 20)
        ),
        "",
    )
    if words:
        found["word_count"] = words
    found = compact(found)
    while found:
        encoded = as_json(found, MAX_LD_CHARS)
        if encoded:
            return encoded
        found.pop(
            max(found, key=lambda k: len(json.dumps(found[k], ensure_ascii=False)))
        )
    return ""


def read_paywalled(nodes: list[dict]) -> bool | None:
    for node in nodes:
        value = node.get("isAccessibleForFree")
        if isinstance(value, str):
            value = (
                value.strip().lower() == "true"
                if value.strip().lower() in ("true", "false")
                else None
            )
        if isinstance(value, bool):
            return not value
    return None
