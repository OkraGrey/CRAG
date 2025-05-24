from ingestion import retriever
from dotenv import load_dotenv
load_dotenv()
import sys
import os
from graph.chains.retrieval_grader import GradeDocument, retrieval_grader
from pprint import pprint
from graph.chains.generation import generation_chain 
# Add the CRAG root directory to sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

def test_retrieval_grader_answer_yes():
    question = "agent memory"
    documents = retriever.invoke(question)
    doc_txt = documents[1].page_content

    res: GradeDocument = retrieval_grader.invoke(
        {"question": question, "documents": doc_txt}
    )

    assert res.binary_score == "yes"
    

def test_retrieval_grader_answer_no():
    question = "agent memory"
    documents = retriever.invoke(question)
    doc_txt = documents[1].page_content

    res: GradeDocument = retrieval_grader.invoke(
        {"question": "How to make a pizza", "documents": doc_txt}
    )

    assert res.binary_score == "no"
    
def test_generation_chain():
    question= "agent Memory"
    docs= retriever.invoke(question)
    generation= generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)