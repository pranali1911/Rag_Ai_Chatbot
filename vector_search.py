from langchain_community.vectorstores import FAISS


# ---------------------------------
# Create FAISS vector store
# ---------------------------------

def create_vector_store(chunks, embedding_model):

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    return vector_store


# ---------------------------------
# Create retriever
# ---------------------------------

def get_retriever(vector_store):

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )

    return retriever