import os

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env if present
load_dotenv(find_dotenv(), override=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def ingest_pdf() -> None:
    pdf_path = os.getenv("PDF_PATH")
    if not pdf_path:
        raise ValueError("PDF_PATH is not set")
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    embedding_model = os.getenv(
        "OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")

    print(f"Loading PDF from {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = splitter.split_documents(documents)
    print(f"Generated {len(splits)} chunks")

    print("Creating embeddings with OpenAI...")
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
    )

    print("Persisting embeddings to Postgres (pgvector) via PGVector store...")
    PGVector.from_documents(
        documents=splits,
        embedding=embeddings,
        connection=database_url,
        collection_name=collection_name,
        pre_delete_collection=False,
    )

    print(
        f"Ingestion completed successfully. Stored in collection '{collection_name}'."
    )


if __name__ == "__main__":
    ingest_pdf()
