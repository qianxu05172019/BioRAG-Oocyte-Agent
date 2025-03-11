from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

class RAGPipeline:
    def __init__(self, persist_directory="VectorStore"):
        # 初始化embedding模型
        self.embeddings = OpenAIEmbeddings()

        # 初始化Chroma向量数据库
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

        # 初始化记忆功能
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # 初始化ConversationalRetrievalChain，加入memory
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0),
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )

    def ask(self, query: str):
        response = self.qa_chain({"question": query})
        return response

# 使用示例
if __name__ == '__main__':
    rag = RAGPipeline()
    while True:
        user_input = input("Please type your question:")
        if user_input.lower() == 'exit':
            break
        result = rag.ask(user_input)
        print("Answer：", result['answer'])
