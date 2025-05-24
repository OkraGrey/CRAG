from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState



def generate(state: GraphState)-> Dict[str,Any]:
    """
    Takes the graph state and returns the state with generated response from the LLM

    Args:
        state: (dict) containing the question and the relevant documents
    Return:
        state: (dict) with generated response

    Note: This function uses 'context' instead of 'documents' while passing the state to the chain
    Reason: We are using 'rlm' prompt from langchain hub and in that prompt we have two fields
    1- question 
    2- context which actually refers to retrieved documents using RAG
    Link to rlm prompt: https://smith.langchain.com/hub/rlm/rag-prompt
    """

    print("--GENERATE--")
    question= state["question"]
    context= state["documents"]

    generation= generation_chain.invoke(
        {"context": context, "question":question}
    )

    return {"documents": context, "question": question, "generation": generation}