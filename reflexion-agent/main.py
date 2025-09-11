from dotenv import load_dotenv

load_dotenv()

from typing import Annotated, TypedDict

from chains import first_responder, revisor
from langchain_core.messages import ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from tool_executor import execute_tools


MAX_ITERATIONS = 2

DRAFT = "draft"
EXECUTE_TOOLS = "execute tools"
REVISE = "revise"


class State(TypedDict):
    messages: Annotated[list, add_messages]


def first_responder_node(state: State):
    return first_responder.invoke({"messages": state["messages"]})


def execute_tools_node(state: State):
    return execute_tools.invoke({"messages": state["messages"]})


def revise_node(state: State):
    return revisor.invoke({"messages": state["messages"]})


builder = StateGraph(State)
builder.add_node(DRAFT, first_responder_node)
builder.add_node(EXECUTE_TOOLS, execute_tools_node)
builder.add_node(REVISE, revise_node)

builder.set_entry_point(DRAFT)

builder.add_edge(DRAFT, EXECUTE_TOOLS)
builder.add_edge(EXECUTE_TOOLS, REVISE)


def event_loop(state: State):
    count_tool_visits = sum(isinstance(item, ToolMessage)
                            for item in state["messages"])
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return EXECUTE_TOOLS


builder.add_conditional_edges(
    REVISE, event_loop, {END: END, EXECUTE_TOOLS: EXECUTE_TOOLS}
)

graph = builder.compile()

print(graph.get_graph().draw_ascii())

if __name__ == "__main__":
    print("Hello World")
