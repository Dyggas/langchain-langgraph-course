from dotenv import load_dotenv

load_dotenv()

from graph.state import GraphState
from graph.graph import app 

if __name__== "__main__":
    print("Hello RAG")
    initial_state = GraphState(
        question="What is agent memory?",
        generation="",
        web_search=True,
        documents=[],
    )
    print(app.invoke(initial_state))