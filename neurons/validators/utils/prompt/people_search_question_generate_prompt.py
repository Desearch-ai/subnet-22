from neurons.validators.utils.prompts import BasePrompt
from datura.utils import call_openai


user_template = """
Generate people search question prompt following the rules specified on system message.
"""

system_message = """
As a search question generator, your task is to generate question prompt searching for specific people on linkedin

1. Important rules
 - Identify the Profession/Background
 - Specify a Transition or Career Shift (optional)
 - Mention an Industry or Sector
 - Add Geographic Context (optional)
 - Focus on a Specific Stage or Milestone (optional)
 - Mention a Specific Skillset or Education (optional)
 - Use Clear, Concise, and Actionable Phrases

** You don't need to follow all the rules as some rules are optional. You can just pick up some of the rules and generate prompt.**
** Don't include words like 'How can I find ...' or 'Looking for ...' **

2. Output
- Output only the question prompt
"""


class PeopleSearchQuestionGeneratePrompt(BasePrompt):
    def __init__(self):
        super().__init__()
        self.template = user_template

    def get_system_message(self):
        return system_message

    async def get_response(self):
        return await call_openai(
            [
                {
                    "role": "system",
                    "content": self.get_system_message(),
                },
                {
                    "role": "user",
                    "content": self.template,
                },
            ],
            temperature=0.8,
            model="gpt-4o-mini",
        )
