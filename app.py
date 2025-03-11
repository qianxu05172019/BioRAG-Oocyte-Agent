import streamlit as st
from src.document_loader import DocumentProcessor
from src.embeddings import VectorStoreManager
from src.rag_pipeline import RAGPipeline
import os


st.set_page_config(
    page_title="Oocyte Expert",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .citation {
        font-size: 0.8rem;
        color: #666;
        border-left: 3px solid #ccc;
        padding-left: 1rem;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None


with st.sidebar:
    st.title("🧬 Oocyte Expert")
    st.markdown("""
    ### About
    This AI assistant specializes in oocyte maturation research, powered by:
    - Comprehensive literature database
    - Advanced language understanding
    - Real-time citation tracking
    """)
    

    if st.session_state.is_initialized:
        st.success("Knowledge Base: Active ✅")
    else:
        st.warning("Knowledge Base: Loading...")
    

    if st.button("Reset System"):
        st.session_state.chat_history = []
        st.experimental_rerun()


st.title("Oocyte Research Assistant")


if not st.session_state.is_initialized:
    with st.spinner("Initializing knowledge base..."):
        try:
            
            vector_store_manager = VectorStoreManager()
            
           
            try:
                vector_store = vector_store_manager.load_vector_store("data/chroma_db")
                st.session_state.vector_store = vector_store
            except ValueError:
                st.error("Vector store not found. Please process PDF documents first.")
                st.stop()
            
           
            st.session_state.rag_pipeline = RAGPipeline("data/chroma_db")
            st.session_state.is_initialized = True
            
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            st.stop()


for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("View Citations"):
                for citation in message["citations"]:
                    st.markdown(f"*{citation}*")


if prompt := st.chat_input("Ask your question about oocyte research..."):
  
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    

    with st.chat_message("assistant"):
        if not st.session_state.rag_pipeline:
            st.error("System not initialized. Please wait...")
        else:
            with st.spinner("Researching..."):
                try:
                   
                    response = st.session_state.rag_pipeline.ask(prompt)
                    
                  
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "citations": ["More detailed citations will be implemented"]  
                    })
                    
                    
                    st.write(response)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()

with col2:
    if st.button("Export Chat"):
     
        st.info("Export feature coming soon!")
