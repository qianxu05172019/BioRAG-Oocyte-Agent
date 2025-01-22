from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  # 这是一个更小更快的模型
        )

    def create_vector_store(self, documents, persist_directory="data/chroma_db"):
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        return vector_store
