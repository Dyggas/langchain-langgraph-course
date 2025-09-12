from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

tavily_tool = TavilySearch(max_results=5)


def run_answer_queries(answer: str, reflection: dict, search_queries: list[str]):
    """Run queries for AnswerQuestion tool."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

def run_revise_queries(answer: str, reflection: dict, search_queries: list[str], references: list[str]):
    """Run queries for ReviseAnswer tool."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

execute_tools = ToolNode([
    StructuredTool.from_function(
        run_answer_queries, 
        name=AnswerQuestion.__name__,
        args_schema=AnswerQuestion
    ),
    StructuredTool.from_function(
        run_revise_queries, 
        name=ReviseAnswer.__name__,
        args_schema=ReviseAnswer
    ),
])
