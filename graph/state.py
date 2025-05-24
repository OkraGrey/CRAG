from typing import TypedDict, List


class GraphState(TypedDict):
    """
    Represents the state of our graph

    Attributes:
        question: question
        generation: LLM generation
        web_search: Whether to add search
        documents: List of documents
    """
    question: str
    generation: str
    web_search: bool 
    documents: List[str]