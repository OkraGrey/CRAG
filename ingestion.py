from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Loading document urls into Langchain documents
# Below code will fetch the docs and it will create a list of them

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

#Split the text for our RAG system

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=20
)

# Splitting the docs
split_docs = text_splitter.split_documents(docs_list)

# Vector Storage Using ChromaDb

# vector_store = Chroma.from_documents(
#     documents=split_docs,
#     collection_name="rag-chroma",
#     embedding=OpenAIEmbeddings(
#         model="text-embedding-3-small",
#         api_key=os.getenv("OPEN_AI_API_KEY")
#     ),
#     persist_directory="./.chroma",
# )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPEN_AI_API_KEY")
    )
).as_retriever()