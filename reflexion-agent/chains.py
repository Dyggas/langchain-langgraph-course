import datetime

from langchain_core.output_parsers.openai_tools import (JsonOutputToolsParser,
                                                        PydanticToolsParser)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from schemas import AnswerQuestion, ReviseAnswer

llm = AzureChatOpenAI()
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required format.",
        ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """Revise your previous answer using the new information from the search results.
    
CRITICAL REQUIREMENTS:
1. Use the search results provided in the previous messages to improve your answer
2. You MUST extract URLs from the search results and create numbered references
3. Every factual claim should have a citation [1], [2], etc.
4. The References section is MANDATORY - do not skip it under any circumstances

Format references exactly like this:
References:
- [1] https://actual-url-from-search-results.com
- [2] https://another-url-from-search-results.com

SEARCH RESULTS ARE AVAILABLE IN THE CONVERSATION ABOVE - USE THEM TO CREATE REFERENCES.

Additional instructions:
- You should use the previous critique to add important information to your answer.
- You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
