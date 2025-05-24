from typing import Any, Dict
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state:GraphState) -> Dict[str, Any]:
    """
    Determine whether the retireved documents are relevant to the question or not.
    If not, we will set the web_search flag to true
    
    Args: 
        State (dict): The current Graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updates the web_search state
    """

    print("--CHECKING THE RELEVANCE OF THE DOCUMENTS TO THE QUESTION--")

    question= state["question"]
    documents= state["documents"]

    filtered_docs= []
    web_search= False

    for d in documents:
        score= retrieval_grader.invoke(
            {"question":question, "documents": d.page_content}
        )
        grade= score.binary_score
        if grade.lower() == "yes":
            print("--GRADE: DOCUMENT IS RELEVANT--")
            filtered_docs.append(d)
        else:
            print("--GRADE: DOCUMENT IS NOT RELEVANT--")
            web_search= True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}