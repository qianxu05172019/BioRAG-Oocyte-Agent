from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_pdfs(self, directory_path):
        """加载PDF文件夹"""
        loader = DirectoryLoader(
            directory_path, 
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
