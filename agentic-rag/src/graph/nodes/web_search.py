from typing import Any, Dict

from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Perform a web search using the question in the state and update the state with the retrieved documents.

    Args:
        state (GraphState): The current graph state containing the question.

    Returns:
        Dict[str, Any]: Updated state with retrieved documents.
    """
    print("---PERFORMING WEB SEARCH---")
    question = state["question"]
    search_results = web_search_tool.invoke({"query": question})["results"]
    joined_results = "\n".join([result["content"] for result in search_results])
    documents = []
    for result in search_results:
        doc = Document(
            page_content=result["snippet"], metadata={"source": result["link"]}
        )
        documents.append(doc)
    web_results = Document(page_content=joined_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question, "web_search": False}


# if __name__ == "__main__":
#     test_state = GraphState(question="What is prompt engineering?", generation="", web_search=True, documents=[])
#     updated_state = web_search(test_state)
#     print(updated_state)
