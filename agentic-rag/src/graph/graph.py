from dotenv import load_dotenv

from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()

def decide_to_generate(state: GraphState) -> str:
    print("---DECIDING NEXT STEP: ACCESSING DOCUMENTS---")
    if state["web_search"]:
        print("----DECISION: WEB SEARCH TRIGGERED----")
        return WEBSEARCH
    else:
        print("----DECISION: DOCUMENTS ARE RELEVANT, GENERATE ANSWER----")
        return GENERATE

    
workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        GENERATE: GENERATE,
        WEBSEARCH: WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END) 

app = workflow.compile()

# app.get_graph().draw_mermaid_png(output_file_path="graph.png")
