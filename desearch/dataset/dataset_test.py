from dataset import QuestionsDataset
import random


if __name__ == "__main__":
    # Create an instance of QuestionsDataset
    questions_dataset = QuestionsDataset()
    tools = [
        ["Twitter Search", "Web Search", "Reddit Search", "Hacker News Search"],
        ["Twitter Search", "Reddit Search"],
        ["Twitter Search", "Web Search", "Reddit Search", "Hacker News Search"],
        ["Twitter Search", "Web Search"],
        ["Twitter Search", "Hacker News Search"],
        ["Twitter Search", "Web Search", "Wikipedia Search", "ArXiv Search"],
        ["Twitter Search", "Youtube Search", "Wikipedia Search"],
        ["Twitter Search", "Youtube Search"],
        ["Twitter Search", "ArXiv Search"],
        ["Twitter Search", "Wikipedia Search"],
    ]

    # Define the selected tools for generating questions
    selected_tools = random.choice(tools)

    # Define the number of questions to generate
    num_questions_to_generate = 40

    # Generate and print questions asynchronously
    async def generate_and_print_questions():
        for _ in range(num_questions_to_generate):
            question = await questions_dataset.generate_new_question_with_openai(
                selected_tools
            )
            print(question)

    # Run the async function
    import asyncio

    asyncio.run(generate_and_print_questions())
