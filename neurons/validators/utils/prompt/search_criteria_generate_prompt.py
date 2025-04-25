from neurons.validators.utils.prompts import BasePrompt
from datura.utils import call_openai


user_template = """
<UserQuestion>
{}
</UserQuestion>
"""

system_message = """
As a search criteria generator, your task is to convert the user question into structured search criteria by breaking it into distinct, actionable points.

1. Input Information
- User Question: <UserQuestion>

2. Important rules
 - Each criterion has to be clear and actionable, avoiding vague terms. For example, instead of 'worked at a tech company' specify 'worked at a software startup.'.
 - Ensure that each criterion is short and to the point. Avoid unnecessary details or long sentences.
 - Break down the input into logically separate points. Each criterion should represent a unique aspect of the original input.
 - Ensure the criteria closely reflect the key elements of the user's input. Maintain the same focus and avoid adding unrelated details.
 - The criteria should be easy to search for or match in a system, meaning they should be phrased in a way that's directly applicable in searches.
 - Don't make assumptions about details that aren't explicitly mentioned in the input. Only use the information provided.
 - If location is mentioned, be sure to incorporate it into the criteria explicitly, as it can be an important factor in search.
 - Maintain a consistent structure and wording for each criterion to ensure they are easy to read and compare.
 - When applicable, specify the role and context in which the individual works, such as company size, industry, or stage
 - If the input refers to time, make sure to capture those aspects, such as joining at an early stage of the company or project.
 - Ensure each criterion is generated from the key concepts of the input. It should not simply copy and repeat the input wording.

3. Output
- Output only the criteria, each on a new line.

-Example Output (Input: Early engineers at AI unicorns, based in San Francisco)
Engineer role at an AI unicorn company
Joined the company during its early stages (e.g., first 50 employees)
Based in San Francisco
"""


class SearchCriteriaGeneratePrompt(BasePrompt):
    def __init__(self):
        super().__init__()
        self.template = user_template

    def get_system_message(self):
        return system_message

    async def get_response(self, query):
        return await call_openai(
            [
                {
                    "role": "system",
                    "content": self.get_system_message(),
                },
                {
                    "role": "user",
                    "content": self.text(query),
                },
            ],
            temperature=0.8,
            model="gpt-4o-mini",
        )
