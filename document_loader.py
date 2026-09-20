from langchain_community.document_loaders import PyPDFLoader, TextLoader


# Load the document based on the file type
def load_document(file_path):

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith(".txt"):
        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

    else:
        raise ValueError(
            "Only PDF and TXT files are supported."
        )

    documents = loader.load()

    return documents