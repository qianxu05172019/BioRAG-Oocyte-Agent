import os
import streamlit as st
from src.document_loader import DocumentProcessor
from src.embeddings import VectorStoreManager
from src.rag_pipeline import RAGPipeline

# 页面配置
st.set_page_config(
    page_title="Oocyte Expert",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加自定义CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)

# 初始化session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False

# 侧边栏
with st.sidebar:
    st.image("https://raw.githubusercontent.com/yourusername/biorag-oocyte/main/assets/logo.png", width=100)
    st.title("Oocyte Expert")
    st.markdown("""
    👋 Welcome to the Oocyte Expert! This AI assistant is specialized in oocyte maturation research.
    
    📚 Knowledge base includes hundreds of papers about:
    - Oocyte development
    - Maturation mechanisms
    - Clinical applications
    """)
    
    with st.expander("About"):
        st.markdown("""
        This RAG (Retrieval Augmented Generation) system combines:
        - Specialized scientific knowledge
        - Advanced language understanding
        - Real-time citation tracking
        
        Created by [Your Name](https://github.com/yourusername)
        """)

# 初始化RAG系统
if not st.session_state.is_initialized:
    with st.spinner("Initializing knowledge base..."):
        try:
            # 初始化向量存储和RAG pipeline
            vector_store_manager = VectorStoreManager()
            # 这里应该使用预先处理好的向量存储
            vector_store = vector_store_manager.load_vector_store("data/chroma_db")
            st.session_state.rag_pipeline = RAGPipeline(vector_store)
            st.session_state.is_initialized = True
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")

# 主界面
st.title("Ask about Oocyte Research 🧬")

# 显示聊天历史
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "citations" in message:
            with st.expander("View Citations"):
                st.info(message["citations"])

# 用户输入
if prompt := st.chat_input("Ask your question about oocyte research..."):
    if not st.session_state.is_initialized:
        st.error("System not initialized. Please wait...")
    else:
        # 添加用户问题
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })

        # 获取回答
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_pipeline.ask(prompt)
                    message = {
                        "role": "assistant",
                        "content": response,
                        "citations": ["Citation details will be added here"]  # Update with actual citations
                    }
                    st.session_state.chat_history.append(message)
                    st.write(response)
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# 页脚
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

with col2:
    if st.button("Export Chat"):
        # Add export functionality here
        pass

with col3:
    if st.button("Report Issue"):
        st.link_button("Report an Issue", "https://github.com/yourusername/biorag-oocyte/issues")
