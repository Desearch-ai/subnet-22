from openai import AsyncOpenAI
from datura.protocol import ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


def system_message(user_system_message):
    output_example = """
        Key News:
            - [Kobold letters: Why HTML emails are a risk to your organization](https://news.ycombinator.com/item?id=39928558)
            - [SportAccord highlights the success of the Social in the City 2024 event](https://news.ycombinator.com/item?id=39921096)
        Hacker News Summary:
            Georgia, as a country, hosts a diverse range of sports events catering to various interests. Popular sports in Georgia include football, basketball, rugby union, wrestling, judo, and weightlifting. The sports industry in Georgia is thriving, with a growing interest in modern sports like rugby union, weightlifting, basketball, judo, and football. The country offers a wide array of sporting activities from traditional sports like polo to modern events like football matches, showcasing a rich sporting culture.
    """

    return f"""
    As a Hacker News data analyst, your task is to provide users with a clear and concise summary derived from the given Hacker News data and the user's query.

    Output Guidelines (Tasks):
    1. Key News: Provide a selection of Hacker News links that directly correspond to the <UserPrompt>.
    Synthesize insights from both the <UserPrompt> and the <HackerNewsData> to formulate a well-rounded response.
    2. Summarizes key News

    <OutputExample>
    {output_example}
    </OutputExample>

    <HackerNewsSummaryRule>
    {user_system_message}
    </HackerNewsSummaryRule>

    Operational Rules:
    1. No <HackerNewsData> Scenario: If no HackerNewsData is provided, inform the user that current Hacker News insights related to their topic are unavailable.
    2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
    3. Seamless Integration: Avoid explicitly stating "Based on the provided <HackerNewsData>" in responses. Assume user awareness of the data integration process.
    4. Separate your responses into sections for easy reading.
    5. For each link, provide only the title and URL without any additional text between them.
    6. Do not return text like <UserPrompt>, <PromptAnalysis>, <PromptAnalysis> in your response; make the response easy to understand for any user.
    7. Make headers bold using Markdown.
    8. Return up to 10 Hacker News links if available.
    9. Do not number the "Key News"; instead, provide each link starting with "-" on a new line.
    10. Always maintain the order as shown in <OutputExample>, first providing "Key News", followed by "Hacker News Summary".
    
    **Follow the rules on <HackerNewsSummaryRule> for writing "Hacker News Summary".**
    """


async def summarize_hacker_news_data(
    prompt: str, model: str, filtered_posts, user_system_message
):
    content = f"""
    In <UserPrompt> provided User's prompt (Question).
    In <HackerNewsData>, Provided Hacker News API fetched data.

    <UserPrompt>
    {prompt}
    </UserPrompt>

    <HackerNewsData>
    {filtered_posts}
    </HackerNewsData>
    """

    messages = [
        {"role": "system", "content": system_message(user_system_message)},
        {"role": "user", "content": content},
    ]

    res = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        stream=True,
    )

    return res, ScraperTextRole.HACKER_NEWS_SUMMARY


def prepare_hacker_news_data_for_summary(posts):
    pass
