from dotenv import load_dotenv

from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import vector_stores

load_dotenv()

pdf_path = Path(__file__).parent / "assets" / "nodejs.pdf"

VECTOR_STORE_URL = "http://localhost:6333/"

# Load this file
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()  # every page is a doc

# Split the docs into smaller chuncks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
"""
Chunk overlap is what ensure a part from the prev chunk will be reread, to ensure proper context
"""

chunks = text_splitter.split_documents(documents=docs)

# Create vector embaddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Create vector store
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url=VECTOR_STORE_URL,
    collection_name="learning_rag",
)

print("INDEXING OF DOCS DONE")
