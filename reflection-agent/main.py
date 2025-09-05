from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from dotenv import load_dotenv

load_dotenv()

from chains import generate_chain, reflect_chain


REFLECT = "reflect"
GENERATE = "generate"


class State(TypedDict):
    messages: Annotated[list, add_messages]


def generation_node(state: State):
    return generate_chain.invoke({"messages": state["messages"]})
    # return state['messages'].append(generate_chain.invoke({"messages": state["messages"]}))


def reflection_node(state: State):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}
    # return state['messages'].append(HumanMessage(content = res.content))


builder = StateGraph(State)
# builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: State):
    if len(state["messages"]) > 5:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    inputs = HumanMessage(content="Trump died.")
    initial_state: State = {"messages": [inputs]}
    response = graph.invoke(initial_state)
