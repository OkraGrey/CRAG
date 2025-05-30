from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import RETRIEVE,GENERATE, WEBSEARCH, GRADE_DOCUMENTS
from graph.nodes import generate,retrieve,grade_documents,web_search
from graph.state import GraphState

load_dotenv()

def decide_to_generate(state):
    print("--ASSESS GRADED DOCS--")

    if state["web_search"]:
        print("--NOT ALL DOCS ARE RELEVANT TO QUESTION--")
        print("--DECISION: WEB_SEARCH--")
        return WEBSEARCH
    else:
        print("--DECISION: GENERATE--")
        return GENERATE
    
workflow= StateGraph(GraphState)

workflow.add_node(RETRIEVE,retrieve)
workflow.add_node(GENERATE,generate)
workflow.add_node(GRADE_DOCUMENTS,grade_documents)
workflow.add_node(WEBSEARCH,web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE,GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS, 
    decide_to_generate,
)
workflow.add_edge(WEBSEARCH,GENERATE)
workflow.add_edge(GENERATE,END)

app= workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="CRAG_Agent.png")