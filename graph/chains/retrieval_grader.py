from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature="0",
    api_key=os.getenv("OPEN_AI_API_KEY")
)

class GradeDocument(BaseModel):
    """Binary Score for document Relevency"""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(schema=GradeDocument)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved documents : \n\n {documents} \n\n User Question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader