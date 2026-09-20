
from dotenv import load_dotenv

# Load .env before importing rag_graph
load_dotenv()

from document_loader import load_document
from text_splitter import split_documents
from embedding import get_embedding_model
from vector_search import create_vector_store, get_retriever
from rag_graph import create_rag_graph



# ---------------------------------
# 1. Load PDF
# ---------------------------------

documents = load_document(
    "Data/Ebook-Agentic-AI.pdf"
)

print(f"Loaded documents: {len(documents)}")


# ---------------------------------
# 2. Split documents
# ---------------------------------

chunks = split_documents(documents)

print(f"Created chunks: {len(chunks)}")


# ---------------------------------
# 3. Create embedding model
# ---------------------------------

embedding_model = get_embedding_model()


# ---------------------------------
# 4. Create FAISS vector store
# ---------------------------------

vector_store = create_vector_store(
    chunks,
    embedding_model
)


# ---------------------------------
# 5. Create retriever
# ---------------------------------

retriever = get_retriever(vector_store)


# ---------------------------------
# 6. Create LangGraph
# ---------------------------------

rag_graph = create_rag_graph(vector_store)


# ---------------------------------
# 7. Ask question
# ---------------------------------

question = input("\nEnter your question: ")

result = rag_graph.invoke({
    "question": question,
    "context": [],
    "scores": [],
    "answer": ""
})


# ---------------------------------
# 8. Display answer
# ---------------------------------

print("\n" + "=" * 60)
print("QUESTION")
print("=" * 60)

print(question)


print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(result["answer"])


print("\n" + "=" * 60)
print("RETRIEVAL SCORES")
print("=" * 60)

for i, score in enumerate(result["scores"], start=1):
    print(f"Chunk {i}: {score:.4f}")

# ---------------------------------
# 9. Display retrieved context
# ---------------------------------

print("\n" + "=" * 60)
print("RETRIEVED CONTEXT")
print("=" * 60)

for i, document in enumerate(
    result["context"],
    start=1
):

    print(f"\n--- Chunk {i} ---")

    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)
