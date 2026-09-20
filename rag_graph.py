import os
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# ---------------------------------
# RAG State
# ---------------------------------

class RAGState(TypedDict):
    question: str
    context: list
    scores: list
    answer: str


# ---------------------------------
# LLM
# ---------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# ---------------------------------
# Prompt
# ---------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a RAG question-answering assistant.

Answer the user's question using ONLY the provided context
from the Agentic AI eBook.

Rules:
- Do not use outside knowledge.
- Do not make up information.
- Keep the answer clear and concise.
- Prefer information directly stated in the context.

If the answer cannot be found in the provided context, say:

"I could not find this information in the provided Agentic AI eBook."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ---------------------------------
# Retrieve documents + scores
# ---------------------------------

def retrieve_documents(state: RAGState, vector_store):

    question = state["question"]

    results = vector_store.similarity_search_with_score(
        question,
        k=5
    )

    documents = [
        document
        for document, score in results
    ]

    scores = [
        float(score)
        for document, score in results
    ]

    return {
        "context": documents,
        "scores": scores
    }


# ---------------------------------
# Generate answer
# ---------------------------------

def generate_answer(state: RAGState):

    question = state["question"]
    documents = state["context"]

    context_text = "\n\n".join(
        document.page_content
        for document in documents
    )

    formatted_prompt = prompt.invoke({
        "context": context_text,
        "question": question
    })

    response = llm.invoke(formatted_prompt)

    return {
        "answer": response.content
    }


# ---------------------------------
# Create LangGraph
# ---------------------------------

def create_rag_graph(vector_store):

    def retrieve_node(state):
        return retrieve_documents(
            state,
            vector_store
        )

    graph = StateGraph(RAGState)

    graph.add_node(
        "retrieve",
        retrieve_node
    )

    graph.add_node(
        "generate",
        generate_answer
    )

    graph.add_edge(
        START,
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "generate"
    )

    graph.add_edge(
        "generate",
        END
    )

    return graph.compile()