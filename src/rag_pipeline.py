from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class RAGPipeline:
    def __init__(self, vector_store, model_name="gpt-3.5-turbo"):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model_name=model_name)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory
        )
