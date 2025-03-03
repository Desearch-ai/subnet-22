from openai import AsyncOpenAI
from datura.protocol import TwitterPromptAnalysisResult, ScraperTextRole

client = AsyncOpenAI(timeout=60.0)


def system_message(user_system_message):
    output_example = """
        Key Tweets:
            - [Noah discusses how SportAccord can elevate the West Midlands brand globally, emphasizing its role in hosting high-profile sports events.](https://x.com/sportaccord/status/456)
            - [SportAccord highlights the success of the Social in the City 2024 event, where Georgia Tech alumni gathered from across the country to celebrate their community spirit.](https://x.com/sportaccord/status/123)
        Twitter Summary:
            Georgia, as a country, hosts a diverse range of sports events catering to various interests. Popular sports in Georgia include football, basketball, rugby union, wrestling, judo, and weightlifting. The sports industry in Georgia is thriving, with a growing interest in modern sports like rugby union, weightlifting, basketball, judo, and football. The country offers a wide array of sporting activities from traditional sports like polo to modern events like football matches, showcasing a rich sporting culture.
    """

    return f"""
    As a Twitter data analyst, your task is to provide users with a clear and concise summary derived from the given Twitter data and the user's query.

    Output Guidelines (Tasks):
    1. Key tweets: Provide a selection of Twitter links that directly correspond to the <UserPrompt>.
    Synthesize insights from both the <UserPrompt> and the <TwitterData> to formulate a well-rounded response.
    2. Summarizes key tweets

    <OutputExample>
    {output_example}
    </OutputExample>

    <TwitterSummaryRule>
    {user_system_message}
    </TwitterSummaryRule>

    Operational Rules:
    1. No <TwitterData> Scenario: If no TwitterData is provided, inform the user that current Twitter insights related to their topic are unavailable.
    2. Emphasis on Critical Issues: Focus on and clearly explain any significant issues or points of interest that emerge from the analysis.
    3. Seamless Integration: Avoid explicitly stating "Based on the provided <TwitterData>" in responses. Assume user awareness of the data integration process.
    4. Please separate your responses into sections for easy reading.
    5. For each link title, include a concise explanation that connects its relevance to the user's question. Use <TwitterData>.url for generate tweet link, example: [username and explanation](<TwitterData>.url)
    6. Not return text like <UserPrompt>, <PromptAnalysis>, <PromptAnalysis> to your response, make response easy to understand to any user.
    7. Make headers bold using Markdown.
    8. Do not number the "key tweets"; instead, provide each on a new line.
    9. Always maintain the order as shown in <OutputExample>, first providing "Key Tweets", followed by "Twitter Summary".
    10. Always return 10 links if available
    
    **Follow the rules on <TwitterSummaryRule> for writing "Twitter Summary".**
    """


async def summarize_twitter_data(
    prompt: str,
    model: str,
    filtered_tweets,
    prompt_analysis: TwitterPromptAnalysisResult,
    user_system_message,
):

    content = f"""
    In <UserPrompt> provided User's prompt (Question).
    In <PromptAnalysis> I analyze that prompts and generate query for API, keywords, hashtags, user_mentions.
    In <TwitterData>, Provided Twitter API fetched data.

    <UserPrompt>
    {prompt}
    </UserPrompt>

    <TwitterData>
    {filtered_tweets}
    </TwitterData>

    <PromptAnalysis>
    {prompt_analysis}
    </PromptAnalysis>
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

    return res, ScraperTextRole.TWITTER_SUMMARY


def prepare_tweets_data_for_summary(tweets):
    data = []

    users = tweets.get("includes", {}).get("users", [])

    for tweet in tweets.get("data", []):
        author_id = tweet.get("author_id")

        author = (
            next((user for user in users if user.get("id") == author_id), None) or {}
        )

        data.append(
            {
                "id": tweet.get("id"),
                "text": tweet.get("text"),
                "author_id": tweet.get("author_id"),
                "created_at": tweet.get("created_at"),
                "url": "https://x.com/{}/status/{}".format(
                    author.get("username"), tweet.get("id")
                ),
                "username": author.get("username"),
            }
        )

    return data
