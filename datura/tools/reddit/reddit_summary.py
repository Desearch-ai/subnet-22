from openai import AsyncOpenAI
from datura.protocol import ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


def system_message():
    output_example = """
        Key Posts:
            - [Noah discusses how SportAccord can elevate the West Midlands brand globally, emphasizing its role in hosting high-profile sports events.](https://reddit.com/r/subreddit/comments/abc/sport-events)
            - [SportAccord highlights the success of the Social in the City 2024 event, where Georgia Tech alumni gathered from across the country to celebrate their community spirit.](https://reddit.com/r/subreddit/comments/abc/sport-events)
        Reddit Summary:
            Georgia, as a country, hosts a diverse range of sports events catering to various interests. Popular sports in Georgia include football, basketball, rugby union, wrestling, judo, and weightlifting. The sports industry in Georgia is thriving, with a growing interest in modern sports like rugby union, weightlifting, basketball, judo, and football. The country offers a wide array of sporting activities from traditional sports like polo to modern events like football matches, showcasing a rich sporting culture.
    """

    return f"""
    As a Reddit data analyst, your task is to provide users with a clear and concise summary derived from the given Reddit data and the user's query.

    Output Guidelines (Tasks):
    1. Key posts: Provide a selection of Reddit links that directly correspond to the <UserPrompt>.
    Synthesize insights from both the <UserPrompt> and the <RedditData> to formulate a well-rounded response.
    2. Summarizes key posts

    <OutputExample>
    {output_example}
    </OutputExample>

    Operational Rules:
    1. No <RedditData> Scenario: If no RedditData is provided, inform the user that current Reddit insights related to their topic are unavailable.
    2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
    3. Seamless Integration: Avoid explicitly stating "Based on the provided <RedditData>" in responses. Assume user awareness of the data integration process.
    4. Please separate your responses into sections for easy reading.
    5. For each link title, include a concise explanation that connects its relevance to the user's question. Use <RedditData>.url for link
    6. Not return text like <UserPrompt>, <PromptAnalysis>, <PromptAnalysis> to your response, make response easy to understand to any user.
    7. Make headers bold using Markdown.
    8. Return up to 10 Reddit links if available.
    9. Do not number the "key posts"; instead, provide each on a new line.
    10. Always maintain the order as shown in <OutputExample>, first providing "Key Posts", followed by "Reddit Summary".
    """


async def summarize_reddit_data(
    prompt: str,
    model: str,
    filtered_posts,
):
    content = f"""
    In <UserPrompt> provided User's prompt (Question).
    In <RedditData>, Provided Reddit API fetched data.

    <UserPrompt>
    {prompt}
    </UserPrompt>

    <RedditData>
    {filtered_posts}
    </RedditData>
    """

    messages = [
        {"role": "system", "content": system_message()},
        {"role": "user", "content": content},
    ]

    res = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        stream=True,
    )

    return res, ScraperTextRole.REDDIT_SUMMARY


def prepare_reddit_data_for_summary(posts):
    pass
