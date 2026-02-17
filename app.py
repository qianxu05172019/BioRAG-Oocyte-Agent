import streamlit as st
from src.embeddings import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import os

load_dotenv()

# 支持 Streamlit Cloud secrets 和本地 .env 两种方式
if not os.getenv("OPENAI_API_KEY"):
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("Please set OPENAI_API_KEY in .env file or Streamlit Cloud Secrets.")
        st.stop()
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
        st.rerun()


st.title("Oocyte Research Assistant")


if not st.session_state.is_initialized:
    with st.spinner("Initializing knowledge base..."):
        try:
            vector_store_manager = VectorStoreManager()

            # 尝试加载已有向量库，不存在则自动构建
            try:
                vector_store = vector_store_manager.load_vector_store("data/chroma_db")
            except ValueError:
                st.info("Building vector store for the first time, this may take a moment...")
                from src.document_loader import DocumentProcessor
                processor = DocumentProcessor()
                documents = processor.load_pdfs("data/papers")
                if not documents:
                    st.error("No PDF documents found in data/papers/")
                    st.stop()
                vector_store = vector_store_manager.create_vector_store(documents)

            st.session_state.rag_pipeline = RAGPipeline(vector_store)
            st.session_state.is_initialized = True

        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            st.stop()


SUGGESTED_QUESTIONS = {
    "OmniPath": [
        "What is OmniPath and what types of biological data does it integrate?",
        "How does OmniPath compare to other pathway databases like KEGG or Reactome?",
        "What are the main data sources combined in OmniPath?",
    ],
    "CellChat & CellPhoneDB": [
        "What is CellChat and how does it infer cell-cell communication?",
        "How does CellPhoneDB predict ligand-receptor interactions from scRNA-seq data?",
        "What are the differences between CellChat and CellPhoneDB?",
    ],
    "Oocyte Biology": [
        "What metabolites are secreted by cumulus cells during oocyte maturation?",
        "How do cumulus cells influence oocyte developmental competence?",
        "What signaling pathways regulate oocyte maturation?",
    ],
}

# 没有聊天记录时显示推荐问题
if not st.session_state.chat_history:
    st.markdown("### Try asking:")
    for category, questions in SUGGESTED_QUESTIONS.items():
        st.markdown(f"**{category}**")
        for q in questions:
            if st.button(q, key=q):
                st.session_state.pending_question = q
                st.rerun()

for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("View Citations"):
                for citation in message["citations"]:
                    st.markdown(f"*{citation}*")


# 处理推荐问题点击或用户输入
pending = st.session_state.pop("pending_question", None)
typed = st.chat_input("Ask your question about oocyte research...")
prompt = pending or typed

if prompt:
  
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    

    with st.chat_message("assistant"):
        if not st.session_state.rag_pipeline:
            st.error("System not initialized. Please wait...")
        else:
            with st.spinner("Researching..."):
                try:
                   
                    response = st.session_state.rag_pipeline.ask(prompt)
                    answer = response["answer"]
                    sources = response.get("source_documents", [])

                    # 提取引用信息
                    citations = []
                    for doc in sources:
                        src = doc.metadata.get("source", "")
                        page = doc.metadata.get("page", "")
                        citations.append(f"{src}, Page {int(page) + 1}")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })

                    st.write(answer)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

with col2:
    if st.button("Export Chat"):
     
        st.info("Export feature coming soon!")
