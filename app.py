import streamlit as st
from src.document_loader import DocumentProcessor
from src.embeddings import VectorStoreManager
from src.rag_pipeline import RAGPipeline
import os

# 页面配置
st.set_page_config(
    page_title="Oocyte Expert",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
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

# 初始化session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# 侧边栏
with st.sidebar:
    st.title("🧬 Oocyte Expert")
    st.markdown("""
    ### About
    This AI assistant specializes in oocyte maturation research, powered by:
    - Comprehensive literature database
    - Advanced language understanding
    - Real-time citation tracking
    """)
    
    # 知识库状态显示
    if st.session_state.is_initialized:
        st.success("Knowledge Base: Active ✅")
    else:
        st.warning("Knowledge Base: Loading...")
    
    # 添加重置按钮
    if st.button("Reset System"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# 主界面内容
st.title("Oocyte Research Assistant")

# === 知识库初始化部分（新添加）===
if not st.session_state.is_initialized:
    with st.spinner("Initializing knowledge base..."):
        try:
            # 初始化向量存储管理器
            vector_store_manager = VectorStoreManager()
            
            # 检查是否存在预处理的向量存储
            try:
                vector_store = vector_store_manager.load_vector_store("data/chroma_db")
                st.session_state.vector_store = vector_store
            except ValueError:
                st.error("Vector store not found. Please process PDF documents first.")
                st.stop()
            
            # 初始化RAG pipeline
            st.session_state.rag_pipeline = RAGPipeline(vector_store)
            st.session_state.is_initialized = True
            
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            st.stop()

# 显示聊天历史
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("View Citations"):
                for citation in message["citations"]:
                    st.markdown(f"*{citation}*")

# 用户输入处理
if prompt := st.chat_input("Ask your question about oocyte research..."):
    # 添加用户问题到历史
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # 处理回答
    with st.chat_message("assistant"):
        if not st.session_state.rag_pipeline:
            st.error("System not initialized. Please wait...")
        else:
            with st.spinner("Researching..."):
                try:
                    # 使用RAG pipeline获取回答
                    response = st.session_state.rag_pipeline.ask(prompt)
                    
                    # 添加回答到历史
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "citations": ["More detailed citations will be implemented"]  # 将来可以添加实际引用
                    })
                    
                    # 显示回答
                    st.write(response)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# 页脚
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()

with col2:
    if st.button("Export Chat"):
        # TODO: 实现导出功能
        st.info("Export feature coming soon!")
