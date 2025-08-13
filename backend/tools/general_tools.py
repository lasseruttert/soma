import os
from dotenv import load_dotenv
from datetime import datetime

import chromadb

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool

load_dotenv()
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
chroma_search_tool = create_retriever_tool(
    retriever=retriever,
    name="chroma_search",
    description="Search the Chroma vector store database containing user uploaded documents.",
    )