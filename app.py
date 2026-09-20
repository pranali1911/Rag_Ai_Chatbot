import streamlit as st

from document_loader import load_document
from text_splitter import split_documents
from embedding import get_embedding_model
from vector_search import create_vector_store
from rag_graph import create_rag_graph


# ---------------------------------
# Page configuration
# ---------------------------------

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------
# Load RAG system
# ---------------------------------

@st.cache_resource
def load_rag_system():

    pdf_path = "Data/Ebook-Agentic-AI.pdf"

    documents = load_document(pdf_path)

    chunks = split_documents(documents)

    embedding_model = get_embedding_model()

    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    rag_graph = create_rag_graph(vector_store)

    return rag_graph


rag_graph = load_rag_system()


# ---------------------------------
# Session state
# ---------------------------------

if "result" not in st.session_state:
    st.session_state.result = None


# ---------------------------------
# UI
# ---------------------------------

st.title("🤖 Agentic AI RAG Chatbot")

st.write(
    "Ask questions about the Agentic AI eBook. "
    "Answers are generated only from the provided PDF."
)


question = st.text_input(
    "Enter your question:",
    key="question_input"
)


# ---------------------------------
# Buttons
# ---------------------------------

col1, col2 = st.columns(2)

with col1:

    ask_button = st.button(
        "Ask",
        use_container_width=True
    )


with col2:

    clear_button = st.button(
        "Clear",
        use_container_width=True
    )


# ---------------------------------
# Clear
# ---------------------------------

if clear_button:

    st.session_state.result = None

    st.rerun()


# ---------------------------------
# Ask question
# ---------------------------------

# ---------------------------------
# Ask question
# ---------------------------------

if ask_button:

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "🔄 Searching the eBook and generating answer..."
        ):

            result = rag_graph.invoke({
                "question": question,
                "context": [],
                "scores": [],
                "answer": ""
            })

        st.session_state.result = result


# ---------------------------------
# Display result
# ---------------------------------

if st.session_state.result:

    result = st.session_state.result


    # ---------------------------------
    # Final answer
    # ---------------------------------

    st.subheader("Answer")

    st.write(result["answer"])


    # ---------------------------------
    # Retrieval scores
    # ---------------------------------

    st.subheader("Retrieval Scores")

    for i, score in enumerate(
        result["scores"],
        start=1
    ):

        st.write(
            f"Chunk {i}: {score:.4f}"
        )


    # ---------------------------------
    # Retrieved context
    # ---------------------------------

    st.subheader("Retrieved Context")

    for i, document in enumerate(
        result["context"],
        start=1
    ):

        with st.expander(f"Chunk {i}"):

            st.write(document.page_content)

            st.caption(
                f"Source: {document.metadata.get('source', 'Unknown')}"
            )

            st.caption(
                f"Page: {document.metadata.get('page_label', 'Unknown')}"
            )