import random
import re

import bittensor as bt
from faker import Faker

from desearch.protocol import ScoringModel
from desearch.utils import call_scoring_llm

QUESTION_TEMPERATURE = 1.15

QUESTION_ANGLES = [
    "accessibility",
    "adoption barriers",
    "adoption incentives",
    "community response",
    "competition dynamics",
    "competitive advantage",
    "consumer impact",
    "consumer trust",
    "cost reduction",
    "costs and tradeoffs",
    "customer retention",
    "energy demand",
    "environmental impact",
    "expert criticism",
    "funding trends",
    "infrastructure readiness",
    "interoperability",
    "international coordination",
    "legal disputes",
    "legal liability",
    "long-term maintenance",
    "labor impact",
    "market impact",
    "market concentration",
    "measurement challenges",
    "misinformation risks",
    "platform governance",
    "policy response",
    "privacy impact",
    "procurement challenges",
    "product reliability",
    "public sentiment",
    "quality control",
    "regional inequality",
    "regional comparison",
    "regulatory compliance",
    "resource constraints",
    "rural access",
    "safety concerns",
    "scaling challenges",
    "security risks",
    "scientific evidence",
    "small business impact",
    "standards and certification",
    "stakeholder response",
    "supply shortages",
    "supply chain effects",
    "technical limitations",
    "urban implementation",
    "workforce changes",
]

WEB_FALLBACK_TEMPLATES = [
    "How is {topic} being adopted?",
    "What are common challenges in {topic}?",
    "What are key risks in {topic}?",
    "What policies shape {topic}?",
    "Which companies are leading {topic}?",
]

RESEARCH_FALLBACK_TEMPLATES = [
    "How are companies approaching {topic}?",
    "How are experts responding to {topic}?",
    "What debates are shaping {topic}?",
    "What risks are emerging around {topic}?",
    "Which groups are affected by {topic}?",
]

TOPICS = [
    "renewable energy",
    "stock market",
    "artificial intelligence",
    "fashion",
    "space exploration",
    "climate change",
    "nutrition",
    "diet",
    "international politics",
    "movies",
    "entertainment",
    "technology",
    "gadgets",
    "medical research",
    "electric vehicles",
    "software development",
    "education",
    "online learning",
    "sustainable agriculture",
    "economic recovery",
    "psychology",
    "mental health",
    "cybersecurity",
    "data privacy",
    "architecture",
    "design",
    "travel",
    "tourism",
    "USA",
    "startup",
    "entrepreneurship",
    "world issues",
    "music",
    "live performances",
    "film",
    "cinema",
    "sport",
    "fitness",
    "gaming",
    "esports",
    "health",
    "wellness",
    "streaming services",
    "cryptocurrency",
    "blockchain",
    "machine learning",
    "American politics",
    "elections",
    "finance",
    "global politics",
    "diplomacy",
    "Olympics",
    "sports competitions",
    "social media",
    "digital communication",
    "art",
    "culture",
    "healthcare",
    "Nvidia AI",
    "Ukraine",
    "geopolitics",
    "Google",
    "programming",
    "science",
    "research",
    "history",
    "blockchain technology",
    "digital health",
    "coffee culture",
    "lifestyle",
    "economy",
    "financial markets",
    "internet culture",
    "social media trends",
    "indie games",
    "game design",
    "video game development",
    "Bitcoin",
    "digital currency",
    "health technology",
    "robotics",
    "automation",
    "movie industry",
    "tech innovation",
    "venture capital",
    "artists",
    "directors",
    "designers",
    "original series",
    "green technology",
    "data science",
    "adventure",
    "US news",
    "investing",
    "voting",
    "world events",
    "content creation",
    "SpaceX",
    "international relations",
    "cloud services",
    "open-source",
    "AI-generated imagery",
    "digital art",
    "cosmos",
    "clean energy",
    "automotive technology",
    "drones",
    "digital finance",
    "global economy",
    "viral content",
    "Microsoft",
    "Internet of Things",
    "fiscal policy",
    "agricultural technology",
    "innovation",
    "virtual reality",
    "affordable housing",
    "mental health awareness",
    "public transportation",
    "e-commerce",
    "autonomous vehicles",
    "international trade agreements",
    "urban development",
    "quantum computing",
    "global migration patterns",
    "space tourism",
    "Amazon",
    "Twitter",
    "quantum mechanics",
    "augmented reality",
    "smart cities",
    "biotechnology",
    "5G networks",
    "gene editing",
    "smart homes",
    "digital identity",
    "sustainable fashion",
    "circular economy",
    "carbon capture technology",
    "precision agriculture",
    "telemedicine",
    "online education platforms",
    "remote work",
    "digital nomads",
    "plant-based meat alternatives",
    "vertical farming",
    "3D printing",
    "robotics in healthcare",
    "edge computing",
    "digital twins",
    "brain-computer interfaces",
    "decentralized finance",
    "non-fungible tokens",
    "space mining",
    "quantum cryptography",
    "smart materials",
    "green hydrogen",
    "tidal energy",
    "carbon offsetting",
    "regenerative agriculture",
    "precision medicine",
    "personalized nutrition",
    "mental health apps",
    "virtual events",
    "augmented reality shopping",
    "drone delivery",
    "self-driving trucks",
    "hyperloop transportation",
    "digital art galleries",
    "virtual influencers",
    "social media activism",
    "gamification in education",
    "bioplastics",
    "ocean cleanup technology",
    "smart waste management",
    "home gardening",
    "local tourism",
    "DIY crafts",
    "home workouts",
    "budget travel",
    "street food",
    "local markets",
    "community service",
    "public parks",
    "urban cycling",
    "pet care",
    "indoor plants",
    "homemade recipes",
    "thrifting",
    "podcasts",
    "audiobooks",
    "e-books",
    "mobile apps",
    "video blogging",
    "online courses",
    "language learning",
    "yoga",
    "meditation",
    "stress management",
    "time management",
    "personal finance",
    "investment basics",
    "meal planning",
    "food preservation",
    "sustainable living",
    "recycling tips",
    "zero waste lifestyle",
    "minimalist living",
    "handicrafts",
    "digital photography",
    "smartphone videography",
    "social media marketing",
    "freelancing",
    "remote work tools",
    "virtual meetings",
    "cyber hygiene",
    "Brazil",
    "India",
    "China",
    "Germany",
    "France",
    "Italy",
    "Japan",
    "Canada",
    "Australia",
    "South Korea",
    "Russia",
    "Spain",
    "Mexico",
    "Indonesia",
    "Turkey",
    "United Kingdom",
    "Saudi Arabia",
    "Netherlands",
    "Switzerland",
    "Sweden",
    "Argentina",
    "Thailand",
    "South Africa",
    "Egypt",
    "Pakistan",
    "Vietnam",
    "Philippines",
    "Nigeria",
    "Kenya",
    "Chile",
    "Colombia",
    "Peru",
    "Malaysia",
    "Singapore",
    "Israel",
    "Denmark",
    "Finland",
    "Ireland",
    "fintech",
    "gig economy",
    "nanotechnology",
    "biohacking",
    "agritech",
    "edtech",
    "wearable tech",
    "deep learning",
    "cloud computing",
    "lab-grown meat",
    "precision fermentation",
    "floating wind farms",
    "eco-friendly packaging",
    "smart grid technology",
    "regenerative tourism",
    "digital nomad visas",
    "AI-powered education",
    "blockchain supply chain",
    "decentralized social networks",
    "virtual real estate",
    "AI-generated music",
    "circular fashion",
    "insect-based protein",
    "space-based solar power",
    "emotion recognition AI",
    "autonomous construction robots",
]


def _clean_question(question: str | None) -> str:
    if not question:
        return ""

    lines = [line.strip() for line in question.splitlines() if line.strip()]
    if not lines:
        return ""

    value = lines[0].strip().strip('"').strip("'")
    value = value.split("\t", 1)[0]
    value = re.split(r"\b(?:intent|label|category)\s*:", value, maxsplit=1, flags=re.I)[
        0
    ]
    value = re.sub(r"^\s*(?:[-*]\s+|\d+[.)]\s*)", "", value).strip()
    value = re.sub(r"\s+", " ", value)
    if value and value[-1] not in "?!.":
        value = f"{value}?"
    return value.strip().strip('"').strip("'")


class _TopicSampler:
    def __init__(self, topics: list[str]) -> None:
        self._topics = list(topics)
        self._topic_pool: list[str] = []

    @property
    def topics(self) -> list[str]:
        return list(self._topics)

    def next_topic(self) -> str:
        if not self._topics:
            return "world events"

        if not self._topic_pool:
            self._topic_pool = list(self._topics)
            random.shuffle(self._topic_pool)

        return self._topic_pool.pop()


class MockTwitterQuestionsDataset:
    """Exposes the shared topic list. Kept for backward compat with callers
    that access `.topics` directly (e.g. research scripts)."""

    def __init__(self):
        self.topics = list(TOPICS)


class QuestionsDataset:
    """Generates AI Search and Web Search questions via LLM.

    Two modes, controlled by the tool set passed to
    `generate_new_question_with_openai`:
      - ['Web Search'] only  -> answerable general web query (static OK)
      - anything else        -> real-time, multi-source research question
    """

    def __init__(self) -> None:
        self._topic_sampler = _TopicSampler(TOPICS)
        self._topics = self._topic_sampler.topics

    def _random_topic(self) -> str:
        return self._topic_sampler.next_topic()

    def _fallback_question(self, topic: str, selected_tools: list[str]) -> str:
        is_web_only = selected_tools == ["Web Search"]
        templates = (
            WEB_FALLBACK_TEMPLATES if is_web_only else RESEARCH_FALLBACK_TEMPLATES
        )
        return random.choice(templates).format(topic=topic)

    def _build_prompt(
        self,
        topic: str,
        selected_tools: list[str],
        angle: str,
    ) -> str:
        is_web_only = selected_tools == ["Web Search"]

        if is_web_only:
            return (
                f'Generate ONE realistic web-search question about "{topic}".\n'
                f"Use this angle if it fits naturally: {angle}.\n"
                "Requirements:\n"
                "- 8 to 12 words long\n"
                "- A specific topical question a real person would search for\n"
                "- Specific enough to have clear answers in web results\n"
                "- Include a concrete entity, location, policy, product, metric, "
                "or stakeholder when it fits\n"
                "- Plain natural language, no hashtags, no quoted phrases\n"
                "- Do not mention search engines or tool names\n"
                "- Do not default to social media, platform trends, or viral "
                "content unless the topic explicitly requires it\n"
                "- Do NOT include time phrases like 'today', 'this week', "
                "'latest', or a specific year — a date filter is applied "
                "separately, so the question text stays time-agnostic\n"
                "Output ONLY the question. No preamble, no quotes."
            )

        tools_str = ", ".join(selected_tools)
        return (
            f'Generate ONE research question about "{topic}" whose best answer '
            f"benefits from combining these sources: {tools_str}.\n"
            f"Use this angle if it fits naturally: {angle}.\n"
            "Requirements:\n"
            "- 8 to 12 words long — keep it short and punchy\n"
            "- A natural topical question, not static trivia\n"
            "- Specific enough that a search will return relevant results\n"
            "- The source names are evidence channels only; do not turn the "
            "question into a social-media or platform-trend question just "
            "because Twitter, Reddit, or YouTube is available\n"
            "- Include a concrete entity, location, policy, product, metric, "
            "or stakeholder when it fits\n"
            "- Plain natural language, no hashtags\n"
            "- Do not name the source tools in the question\n"
            "- Do NOT include time phrases like 'today', 'this week', "
            "'latest', or a specific year — a date filter is applied "
            "separately, so the question text stays time-agnostic\n"
            "Output ONLY the question. No preamble, no quotes."
        )

    async def generate_new_question_with_openai(
        self,
        selected_tools: list[str],
        model: ScoringModel = ScoringModel.OPENAI_GPT4_1_NANO,
    ) -> str:
        topic = self._random_topic()
        angle = random.choice(QUESTION_ANGLES)
        prompt = self._build_prompt(topic, selected_tools, angle)

        try:
            out = await call_scoring_llm(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=QUESTION_TEMPERATURE,
            )
            return _clean_question(out) or self._fallback_question(
                topic, selected_tools
            )
        except Exception as e:
            bt.logging.error(f"generate_new_question_with_openai failed: {e}")
            return self._fallback_question(topic, selected_tools)


class BasicQuestionsDataset:
    """Generates short X/Twitter search queries.

    Queries are intentionally brief so they still return results when combined
    downstream with filters like date range, min likes, or user filters.
    """

    POPULAR_CRYPTO_KEYWORDS = [
        "bitcoin",
        "btc",
        "eth",
        "ethereum",
        "solana",
        "xrp",
        "dogecoin",
        "doge",
        "cardano",
        "ada",
        "avalanche",
        "avax",
        "matic",
        "link",
    ]

    # High-volume accounts that reliably post; safe for `from:` filters.
    POPULAR_ACCOUNTS = [
        "elonmusk",
        "Google",
        "Tesla",
        "NatGeo",
        "MrBeast",
        "binance",
        "coinbase",
        "coindesk",
        "cointelegraph",
        "OpenAI",
        "nvidia",
        "SpaceX",
    ]

    def __init__(self):
        self.faker = Faker()
        self._topic_sampler = _TopicSampler(TOPICS)
        self._topics = self._topic_sampler.topics

    def generate_random_x_query(self) -> str:
        """
        Short Twitter search query.

        Modes:
          - 20%  from:<account>     (single-token, no extra terms)
          - 15%  $<ticker>          (faker-generated crypto cashtag)
          - 15%  #<crypto>          (known crypto hashtag)
          - 50%  <topic>            (plain topic from the seed list)
        """
        mode = random.random()

        if mode < 0.20:
            account = random.choice(self.POPULAR_ACCOUNTS)
            return f"from:{account}"

        if mode < 0.35:
            try:
                code, _ = self.faker.cryptocurrency()
                return f"${code}"
            except Exception:
                pass

        if mode < 0.50:
            kw = random.choice(self.POPULAR_CRYPTO_KEYWORDS)
            return f"#{kw}"

        return self._topic_sampler.next_topic()
