# Agentic AI RAG Chatbot

A simple **Retrieval-Augmented Generation (RAG) chatbot** built in Python.

This project answers questions based only on the provided **Agentic AI eBook**. It uses **LangGraph** to manage the RAG workflow, **FAISS** as the vector database, **HuggingFace embeddings** for text embeddings, **Google Gemini** for answer generation, and **Streamlit** for the user interface.

---

## 1. Project Overview

The goal of this project is to build an AI chatbot that can understand questions about Agentic AI and provide answers from a given PDF knowledge base.

Instead of asking the LLM to answer from its general knowledge, this project first searches the Agentic AI eBook for relevant information.

The basic process is:

```text
User Question
      ↓
Create Question Embedding
      ↓
Search FAISS Vector Database
      ↓
Retrieve Relevant Chunks
      ↓
Pass Retrieved Context to LLM
      ↓
Generate Answer
      ↓
Display Answer + Context + Scores
```

This approach helps the chatbot answer questions using the information available in the provided document.

---

# 2. Knowledge Base

The chatbot uses the following Agentic AI eBook as its knowledge base:

**Agentic AI eBook**

https://konverge.ai/pdf/Ebook-Agentic-AI.pdf

The PDF is loaded, divided into smaller text chunks, converted into embeddings, and stored in the FAISS vector database.

---

# 3. Main Features

The project includes the following features:

* PDF document loading
* Text extraction from the PDF
* Text chunking
* HuggingFace text embeddings
* FAISS vector database
* Similarity-based document retrieval
* LangGraph RAG workflow
* Gemini 2.5 Flash for answer generation
* Streamlit web interface
* Retrieved context display
* Retrieval score display
* Source and page information
* Clear button
* Loading indicator while generating an answer
* Grounded answers based on the provided PDF

---

# 4. Technologies Used

| Technology    | Purpose                                 |
| ------------- | --------------------------------------- |
| Python        | Main programming language               |
| LangChain     | Document processing and LLM integration |
| LangGraph     | RAG workflow management                 |
| HuggingFace   | Text embedding model                    |
| FAISS         | Vector database / similarity search     |
| Google Gemini | Answer generation                       |
| Streamlit     | Web-based user interface                |
| PyPDF         | PDF document loading                    |
| python-dotenv | Loading API key from `.env`             |

---

# 5. Why RAG?

RAG stands for **Retrieval-Augmented Generation**.

A normal LLM can answer questions using the knowledge it learned during training. However, for this project, the chatbot needs to answer questions specifically from the Agentic AI eBook.

RAG solves this by adding a retrieval step before generating the answer.

For example:

```text
User:
What is Agentic AI?

        ↓

Retriever searches the eBook

        ↓

Relevant sections are retrieved

        ↓

Retrieved text is given to Gemini

        ↓

Gemini generates the answer
```

This allows the answer generation step to use information from the provided knowledge base.

---

# 6. Project Architecture

The complete architecture is:

```text
                 Agentic AI PDF
                       |
                       v
                PDF Document Loader
                       |
                       v
                 Text Splitter
                       |
                       v
                  Text Chunks
                       |
                       v
             HuggingFace Embeddings
                       |
                       v
                FAISS Vector Store
                       |
                       v
                  LangGraph
                       |
             +---------+---------+
             |                   |
             v                   |
       Retrieve Top-5            |
       Relevant Chunks           |
             |                   |
             +---------+---------+
                       |
                       v
                Gemini 2.5 Flash
                       |
                       v
                 Final Answer
                       |
                       v
                 Streamlit UI
```

---

# 7. RAG Pipeline

The RAG pipeline has two main stages:

## Stage 1: Document Indexing

This stage prepares the PDF for searching.

```text
PDF
 ↓
Load Document
 ↓
Split into Chunks
 ↓
Generate Embeddings
 ↓
Store in FAISS
```

## Stage 2: Question Answering

This stage runs when the user asks a question.

```text
Question
 ↓
Search FAISS
 ↓
Retrieve Top-5 Chunks
 ↓
Create Context
 ↓
Send Context + Question to Gemini
 ↓
Generate Answer
```

---

# 8. Step-by-Step Working

## Step 1: Load the PDF

The project uses `PyPDFLoader` to load the Agentic AI eBook.

File:

```text
document_loader.py
```

The loader reads the PDF and converts its pages into documents.

Example:

```python
documents = load_document(
    "Data/Ebook-Agentic-AI.pdf"
)
```

---

## Step 2: Split the Document

Large documents are divided into smaller chunks.

File:

```text
text_splitter.py
```

The project uses:

```text
RecursiveCharacterTextSplitter
```

Current configuration:

```text
Chunk Size: 500 characters
Chunk Overlap: 100 characters
```

### Why chunking?

Sending the complete PDF to the LLM for every question would not be efficient.

Instead, the document is divided into smaller pieces so that the retriever can find only the relevant information.

The overlap also helps preserve information that may be split between two chunks.

---

# 9. Text Embeddings

File:

```text
embedding.py
```

The project uses the HuggingFace model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

An embedding converts text into a numerical vector.

For example:

```text
"Agentic AI uses autonomous agents"
              ↓
       Numerical Vector
```

Similar text produces vectors that are closer together in the vector space.

This allows FAISS to find text that is semantically related to the user's question.

---

# 10. FAISS Vector Database

File:

```text
vector_search.py
```

FAISS is used as the vector database for this project.

FAISS stores the embeddings of the document chunks.

When the user asks a question:

```text
Question
   ↓
Question Embedding
   ↓
FAISS Similarity Search
   ↓
Top 5 Relevant Chunks
```

The project uses similarity search to retrieve the top 5 relevant chunks.

FAISS is used instead of Pinecone because the assignment allows:

> Pinecone or any Vector DB of your choice.

FAISS is a local vector search library and is suitable for this project.

---

# 11. LangGraph RAG Workflow

File:

```text
rag_graph.py
```

LangGraph is used to create the RAG workflow.

The workflow contains two main nodes:

```text
START
  ↓
Retrieve
  ↓
Generate
  ↓
END
```

### Retrieve Node

The retrieve node searches the FAISS vector store.

It retrieves the top 5 relevant document chunks.

It also returns the retrieval scores.

### Generate Node

The generate node receives:

* User question
* Retrieved document chunks

These are passed to Gemini 2.5 Flash.

The model generates the final answer using the retrieved context.

---

# 12. Grounded Answer Generation

One important requirement of the project is that the chatbot should answer based only on the provided PDF.

The generation prompt contains instructions such as:

```text
Use only the provided context from the Agentic AI eBook.

Do not use outside knowledge.

Do not make up information.

If the answer cannot be found in the provided context,
say:

"I could not find this information in the provided
Agentic AI eBook."
```

This makes the generation step focused on the retrieved information.

The chatbot is therefore designed to avoid answering questions using unrelated outside knowledge.

---

# 13. LLM

The project uses:

```text
Google Gemini 2.5 Flash
```

The LLM receives:

```text
Retrieved Context
+
User Question
```

and generates the final response.

The API key is stored in a `.env` file instead of directly writing it inside the Python code.

Example:

```text
GOOGLE_API_KEY=your_api_key
```

The `.env` file is excluded from GitHub using `.gitignore`.

---

# 14. Streamlit User Interface

File:

```text
app.py
```

Streamlit is used to create the web interface.

The UI contains:

* Application title
* Question input box
* Ask button
* Clear button
* Loading indicator
* Final answer
* Retrieval scores
* Retrieved context
* Source information
* Page information

When the user clicks **Ask**, the application shows a loading message while the RAG pipeline is running.

Example:

```text
🔄 Searching the eBook and generating answer...
```

After processing is complete, the final answer is displayed.

---

# 15. Retrieved Context

The application also displays the document chunks used to generate the answer.

For example:

```text
Retrieved Context

Chunk 1
Chunk 2
Chunk 3
Chunk 4
Chunk 5
```

Each chunk can be expanded to view its content.

The application also displays source and page information when available.

This makes it easier to understand which parts of the document were retrieved for the question.

---

# 16. Retrieval Scores

The application displays the retrieval score for each retrieved chunk.

Example:

```text
Chunk 1: 0.3665
Chunk 2: 0.4199
Chunk 3: 0.4332
Chunk 4: 0.4748
Chunk 5: 0.4917
```

These are **FAISS retrieval distance scores**.

They are not confidence percentages.

For the current FAISS search:

```text
Lower distance = closer match
Higher distance = less similar match
```

The scores are included because the assignment requires a confidence or retrieval score.

---

# 17. Project Structure

```text
Rag_Ai/
│
├── Data/
│   └── Ebook-Agentic-AI.pdf
│
├── document_loader.py
├── embedding.py
├── text_splitter.py
├── vector_search.py
├── rag_graph.py
├── test_rag.py
├── app.py
│
├── requirements.txt
├── .gitignore
├── README.md
│
├── .env
└── venv/
```

### File Description

| File                 | Purpose                                  |
| -------------------- | ---------------------------------------- |
| `document_loader.py` | Loads the PDF                            |
| `embedding.py`       | Creates the HuggingFace embedding model  |
| `text_splitter.py`   | Splits documents into chunks             |
| `vector_search.py`   | Creates FAISS vector store and retriever |
| `rag_graph.py`       | Defines the LangGraph RAG workflow       |
| `test_rag.py`        | Tests the RAG pipeline from the terminal |
| `app.py`             | Streamlit web application                |
| `requirements.txt`   | Python dependencies                      |
| `.gitignore`         | Files that should not be uploaded        |
| `README.md`          | Project documentation                    |
| `.env`               | Stores API key locally                   |

---

# 18. Installation and Setup

## Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project folder:

```bash
cd Rag_Ai
```

---

## Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

---

## Step 3: Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

---

## Step 4: Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

---

# 19. Environment Variables

Create a file named:

```text
.env
```

in the project root.

Add your Google Gemini API key:

```text
GOOGLE_API_KEY=your_google_api_key
```

The `.env` file should **never be uploaded to GitHub**.

The project `.gitignore` contains:

```text
.env
venv/
__pycache__/
*.pyc
.streamlit/
```

---

# 20. Run the Application

After activating the virtual environment, run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL such as:

```text
http://localhost:8501
```

Open the URL in a browser.

---

# 21. Testing the RAG Pipeline

The RAG pipeline can also be tested without the Streamlit interface.

Run:

```bash
python test_rag.py
```

The terminal will ask:

```text
Enter your question:
```

Enter a question about the Agentic AI eBook.

The terminal displays:

* Question
* Final answer
* Retrieval scores
* Retrieved context
* Metadata

---

# 22. Sample Queries

The following questions can be used to test the chatbot.

### Query 1

```text
What is Agentic AI?
```

### Query 2

```text
What are the building blocks of an AI agent?
```

### Query 3

```text
What is the difference between Agentic AI and traditional AI?
```

### Query 4

```text
What role does the environment play in an AI agent?
```

### Query 5

```text
What are the characteristics of Agentic AI?
```

### Query 6

```text
How do AI agents interact with their environment?
```

---

# 23. Example Output

For a question such as:

```text
What is Agentic AI?
```

the application returns a final answer generated from the retrieved eBook content.

It also shows:

```text
Retrieval Scores

Chunk 1: 0.3665
Chunk 2: 0.4199
Chunk 3: 0.4332
Chunk 4: 0.4748
Chunk 5: 0.4917
```

The retrieved chunks can be expanded to inspect the information used by the RAG pipeline.

---

# 24. Important Design Decisions

## Why FAISS?

The assignment allows any vector database.

FAISS was selected because:

* It is open source.
* It is easy to use with Python.
* It works well with LangChain.
* It supports similarity search.
* It does not require an external database server for this project.

---

## Why HuggingFace Embeddings?

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

because it provides a lightweight sentence embedding model suitable for semantic search.

---

## Why LangGraph?

LangGraph provides a clear workflow for the RAG pipeline.

The current workflow is:

```text
START
 ↓
Retrieve
 ↓
Generate
 ↓
END
```

This keeps retrieval and answer generation as separate steps.

---

## Why Streamlit?

The assignment allows Streamlit as an alternative to FastAPI/Flask.

Streamlit provides a simple interface for testing the RAG chatbot without building a separate frontend.

---

# 25. Error Handling / Fallback

If the required information is not found in the retrieved context, the chatbot is instructed to return:

```text
I could not find this information in the provided Agentic AI eBook.
```

This helps prevent the chatbot from generating an answer when the information is not available in the provided knowledge base.

---

# 26. Security

The Google API key is stored in:

```text
.env
```

and not directly inside the source code.

The `.env` file is included in `.gitignore`.

API keys should not be committed to the public GitHub repository.

---

# 27. Assignment Requirements Mapping

| Assignment Requirement   | Implementation                      |
| ------------------------ | ----------------------------------- |
| Ingest PDF               | `document_loader.py`                |
| Chunk text               | `text_splitter.py`                  |
| Generate embeddings      | `embedding.py`                      |
| Vector database          | FAISS                               |
| RAG pipeline             | LangGraph                           |
| Retrieve relevant chunks | FAISS similarity search             |
| Generate answer          | Gemini 2.5 Flash                    |
| Grounded response        | Context-based generation prompt     |
| Chat UI                  | Streamlit                           |
| Final answer             | Streamlit output                    |
| Retrieved context        | Streamlit retrieved context section |
| Score                    | FAISS retrieval distance            |
| Sample queries           | 6 sample queries                    |
| Setup instructions       | This README                         |
| Architecture explanation | Architecture section                |

---

# 28. Limitations

This project has some practical limitations:

* The current knowledge base is limited to the provided Agentic AI eBook.
* The vector store is created locally using FAISS.
* The chatbot depends on the Gemini API for answer generation.
* Retrieval scores are distance scores and are not calibrated confidence percentages.
* The chatbot's answer quality depends on the quality of retrieved document chunks.

---

# 29. Future Improvements

Possible future improvements include:

* Using Pinecone for a hosted vector database
* Adding conversation history
* Adding chat memory
* Improving retrieval with hybrid search
* Adding a reranking step
* Adding citation references to specific pages
* Adding evaluation metrics for RAG performance
* Adding a larger or multiple-document knowledge base
* Deploying the Streamlit application online

---

# 30. Conclusion

This project demonstrates a complete RAG pipeline using Python.

The system takes the Agentic AI eBook, converts it into searchable vector representations, retrieves relevant information for a user's question, and uses Gemini to generate an answer from that retrieved context.

The project combines:

```text
Python
+
LangChain
+
LangGraph
+
HuggingFace Embeddings
+
FAISS
+
Gemini
+
Streamlit
```

to create a simple and practical Agentic AI knowledge-base chatbot.

---

## Author

**Pranali Rahangdale**

MCA | AI/ML | Data Science | Generative AI
