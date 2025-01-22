from langchain.chat_models import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

class RAGPipeline:
    def __init__(self, vector_store):
        """Initialize the RAG pipeline with a vector store"""
        load_dotenv()  # Load environment variables
        
        # Use HuggingFace model instead of OpenAI
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",  # 或选择其他合适的模型
            model_kwargs={"temperature": 0.5, "max_length": 512},
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize the conversation chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )
    
    def ask(self, question: str) -> str:
        """Ask a question and get a response"""
        try:
            response = self.chain({"question": question})
            return response['answer']
        except Exception as e:
            return f"Error generating response: {str(e)}"
