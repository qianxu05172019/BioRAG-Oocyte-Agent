from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

class VectorStoreManager:
    def __init__(self):
        load_dotenv()
        # Set the API key as an environment variable
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        # Initialize embeddings without explicit API key parameter
        self.embeddings = OpenAIEmbeddings()

    def create_vector_store(self, documents, persist_directory="data/chroma_db"):
        """Create or update vector store"""
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        vector_store.persist()
        return vector_store

    def load_vector_store(self, persist_directory="data/chroma_db"):
        """Load existing vector store"""
        if not os.path.exists(persist_directory):
            raise ValueError("Vector store not found!")

        return Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
