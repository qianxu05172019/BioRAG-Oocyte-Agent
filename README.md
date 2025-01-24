# BioRAG: Intelligent Research Assistant for Oocyte Studies

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://biorag-oocyte-36nfepumrpgfwushlci6c2.streamlit.app/)
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/qianxu05172019/biorag-oocyte)

## Overview

BioRAG is a specialized Retrieval-Augmented Generation (RAG) system designed to assist researchers in oocyte studies. It combines advanced NLP technologies with scientific literature processing to provide intelligent research assistance.

## Features

- 🔍 Semantic search across scientific papers
- 💬 Interactive research-focused chat interface
- 📚 Real-time citation tracking
- 🎨 Intuitive user interface
- 📊 Persistent session management
- 🔄 System reset functionality

## Technical Architecture

### Core Components

- **RAG Pipeline**: Built with LangChain for efficient information retrieval
- **Embeddings**: OpenAI embeddings for semantic search
- **Document Processing**: Custom PDF processor for scientific literature
- **Language Model**: GPT-3.5-turbo integration
- **Vector Database**: ChromaDB for similarity search
- **Interface**: Streamlit-based web application

### System Architecture

```mermaid
flowchart LR
    docs[Documents] --> loader[DocumentLoader]
    loader --> splitter[TextSplitter]
    splitter --> embeddings[OpenAIEmbeddings]
    embeddings --> chroma[ChromaDB]
    
    query[User Query] --> retriever[Retriever]
    chroma --> retriever
    retriever --> chain[ConversationalRetrievalChain]
    
    memory[ConversationMemory] --> chain
    llm[ChatOpenAI] --> chain
    chain --> response[Response]

    style docs fill:#f9d5e5
    style chroma fill:#eeac99
    style llm fill:#84b6f4
    style response fill:#77dd77
```

### Project Structure

```
project/
├── app.py                 # Streamlit application
├── process_pdfs.py        # PDF processing
├── src/
│   ├── document_loader.py # Document processing
│   ├── embeddings.py      # Vector embeddings
│   └── rag_pipeline.py    # RAG implementation
```

## Deployment

### Cloud Deployment

The application is deployed on Streamlit Cloud with:
- Automated GitHub-based deployment
- Secure environment variable management
- Continuous availability
- Protected API key handling

### Local Development

#### Prerequisites

- Python 3.8+
- OpenAI API key
- Dependencies: streamlit, langchain, chromadb, openai

#### Installation

```bash
git clone https://github.com/qianxu05172019/biorag-oocyte.git
cd biorag
pip install -r requirements.txt
```

#### Configuration

1. Create `.env` in project root
2. Add API key:
```
OPENAI_API_KEY=your-api-key
```

#### Running Locally

```bash
streamlit run app.py
```

## Future Development

### Planned Features

1. **Enhanced Citations**
   - Detailed tracking system
   - Export functionality
   - Citation network visualization

2. **Analytics Integration**
   - Research trend analysis
   - Document clustering
   - Knowledge graph visualization

3. **System Enhancements**
   - Multi-model support (GPT-4, Claude)
   - Automated metadata extraction
   - Enhanced conversation memory
   - Chat history export

4. **Knowledge Base Updates**
   - Meeting notes integration
   - Experiment results tracking
   - Oocyte stage imaging

## Live Demo

Access the live application: [BioRAG Oocyte Expert](https://biorag-oocyte-36nfepumrpgfwushlci6c2.streamlit.app/)

## Contributing

We welcome contributions! Please submit pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
*Developed as a showcase of Machine Learning Engineering and Data Science capabilities, with focus on NLP, RAG systems, and LLM integration.*