from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

llm= ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPEN_AI_API_KEY")
)

prompt= hub.pull("rlm/rag-prompt")
generation_chain= prompt | llm | StrOutputParser()