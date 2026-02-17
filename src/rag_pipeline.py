from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain


class RAGPipeline:
    def __init__(self, vector_store):
        # 接收外部传入的向量库实例，避免重复创建
        self.vector_store = vector_store

        # 初始化记忆功能
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # 初始化ConversationalRetrievalChain，加入memory
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0),
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            return_source_documents=True,
            output_key="answer",
            verbose=True
        )

    def ask(self, query: str):
        response = self.qa_chain({"question": query})
        return response

# 使用示例
if __name__ == '__main__':
    from src.embeddings import VectorStoreManager
    vector_store_manager = VectorStoreManager()
    vector_store = vector_store_manager.load_vector_store()
    rag = RAGPipeline(vector_store)
    while True:
        user_input = input("Please type your question:")
        if user_input.lower() == 'exit':
            break
        result = rag.ask(user_input)
        print("Answer：", result['answer'])
