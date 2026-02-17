from src.document_loader import DocumentProcessor
from src.embeddings import VectorStoreManager
import os

def main():
    # 设置PDF目录路径
    pdf_directory = "data/papers"  # PDF论文存放目录
    
    # 检查目录是否存在
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        print(f"Created directory {pdf_directory}")
        print(f"Please place your PDF files in {pdf_directory} directory and run this script again.")
        return
    
    # 检查目录中是否有PDF文件
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}!")
        print("Please add some PDF files to this directory and run this script again.")
        return
    
    print(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
    
    # 加载并处理PDF文件
    print("Processing PDF documents...")
    document_processor = DocumentProcessor()
    documents = document_processor.load_pdfs(pdf_directory)
    
    if not documents:
        print("Error: No document chunks were generated!")
        return
    
    print(f"Successfully processed {len(documents)} document chunks.")
    
    # 创建向量存储
    print("Creating vector store (this may take a while)...")
    try:
        vector_store_manager = VectorStoreManager()
        vector_store = vector_store_manager.create_vector_store(documents)
        print("Vector store created successfully!")
        print("You can now run the Streamlit app and query your documents.")
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        print("Check your OpenAI API key and ensure it's correctly set in your .env file.")

if __name__ == "__main__":
    main()

