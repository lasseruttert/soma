import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_core.documents import Document

load_dotenv()
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

class ChromaDB:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        self.vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=self.embeddings)
    
    def ingest_documents(self, filepaths):
        docs = []
        for path in filepaths:
            if path.endswith(".txt"):
                loader = TextLoader(path)
            elif path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".csv"):
                loader = CSVLoader(path)
            else:
                continue
            docs.extend(loader.load())
        self.vectorstore.add_documents(docs)

if __name__ == "__main__":
    db = ChromaDB()
    db.ingest_documents(["G:/Meine Ablage/soma/data/documents/example.txt"])