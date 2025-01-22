

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 可以调整这个大小
            chunk_overlap=200,
            length_function=len
        )

    def load_pdfs(self, directory_path):
        """加载指定目录下的所有PDF"""
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",  # 递归搜索所有PDF
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
