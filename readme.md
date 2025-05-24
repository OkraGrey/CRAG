1- First run the ingestion.py file and uncomment the following code.
    vector_store = Chroma.from_documents(
    documents=split_docs,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPEN_AI_API_KEY")
    ),
    persist_directory="./.chroma",
    )

    This will initialize the chromaDB