from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

class VectorStoreManager:
    def __init__(self):
        # 使用 MiniLM 作为嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  # 小型但效果不错的模型
        )

    def create_vector_store(self, documents, persist_directory="data/chroma_db"):
        """创建或更新向量存储"""
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
        """加载已存在的向量存储"""
        if not os.path.exists(persist_directory):
            raise ValueError("Vector store not found!")
            
        return Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
