from src.document_loader import DocumentProcessor
from src.embeddings import VectorStoreManager
import os

def process_all_pdfs():

    doc_processor = DocumentProcessor()
    vector_store_manager = VectorStoreManager()
    

    pdf_dir = "data/papers/"
    db_dir = "data/chroma_db"
    

    print("Loading PDFs...")
    documents = doc_processor.load_pdfs(pdf_dir)
    print(f"Loaded {len(documents)} document chunks")
    

    print("Creating vector store...")
    vector_store = vector_store_manager.create_vector_store(documents, 
db_dir)
    print("Vector store created and persisted")

if __name__ == "__main__":
    process_all_pdfs()


