from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

class RAGPipeline:
    def __init__(self, vector_store, model_name="gpt-3.5-turbo"):
        load_dotenv()  # Load environment variables
        
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory
        )
    
    def ask(self, question: str) -> str:
        """Ask a question and get a response"""
        try:
            response = self.qa_chain({"question": question})
            return response['answer']
        except Exception as e:
            return f"Error generating response: {str(e)}"
