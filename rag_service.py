import os
from pathlib import Path
from dotenv import load_dotenv

# Load API keys from explicit .env path
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class FinancialRAGService:
    def __init__(self):
        # Initialize Google Embeddings using GEMINI_API_KEY
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            api_key=os.getenv("GEMINI_API_KEY") or "placeholder_key"
        )
        # Initialize persistent Chroma vector database
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            collection_name="financial_guidelines",
            persist_directory="./financial_vector_db"
        )

    def initialize_knowledge_base(self):
        """Indexes regulatory policies and guidelines into vector store."""
        docs = [
            Document(
                page_content="SEBI Compliance: AI Financial Assistants must never guarantee market returns or give direct buy/sell mandates. All insights are purely educational and analytical. Risk warnings must accompany all reports.",
                metadata={"category": "compliance"}
            ),
            Document(
                page_content="RSI Interpretation: An RSI above 70 indicates an overbought situation, suggesting caution. An RSI below 30 indicates an oversold asset, suggesting potential reversal.",
                metadata={"category": "technical"}
            ),
            Document(
                page_content="Indian Market Timings: Standard equity trading on NSE and BSE runs from 9:15 AM to 3:30 PM IST on weekdays.",
                metadata={"category": "operations"}
            )
        ]
        # Split documents into smaller chunks for vector retrieval
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        self.vector_store.add_documents(chunks)

    def get_retriever(self):
        # Return retriever to search top 2 matching documents
        return self.vector_store.as_retriever(search_kwargs={"k": 2})

if __name__ == "__main__":
    rag = FinancialRAGService()
    try:
        rag.initialize_knowledge_base()
        print("Vector DB initialized successfully.")
    except Exception as e:
        print(f"Notice: RAG initialization note: {e}")
