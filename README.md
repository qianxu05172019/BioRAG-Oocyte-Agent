# BioRAG: Intelligent Research Assistant for Oocyte Studies

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://biorag-oocyte-36nfepumrpgfwushlci6c2.streamlit.app/)

## Project Overview

BioRAG is a sophisticated Retrieval-Augmented Generation (RAG) system designed to assist researchers in the field of oocyte studies. This project demonstrates the practical application of cutting-edge Machine Learning and Natural Language Processing technologies in biological research.

## Technical Implementation

### Core Technologies
- **RAG Pipeline**: Implemented using LangChain framework for efficient information retrieval and generation
- **Vector Embeddings**: Utilizing OpenAI embeddings for semantic search capabilities
- **Document Processing**: Custom document processor for handling scientific PDFs
- **LLM Integration**: Leveraging GPT-3.5-turbo for natural language understanding and generation
- **Vector Store**: ChromaDB for efficient similarity search and document retrieval
- **Web Interface**: Built with Streamlit for an intuitive user experience

### Architecture
```
project/
├── app.py                 # Main Streamlit application
├── process_pdfs.py        # PDF processing script
├── src/
│   ├── document_loader.py # Document processing module
│   ├── embeddings.py      # Vector embeddings management
│   └── rag_pipeline.py    # RAG implementation
```

## Features

### Current Functionality
- 🔍 Semantic search across scientific literature
- 💬 Interactive chat interface for research queries
- 📚 Real-time citation tracking
- 🎨 Clean, user-friendly interface
- 📊 Session state management
- 🔄 System reset capabilities

### Technical Highlights
- Efficient document chunking with controlled overlap
- Persistent vector storage system
- Conversation memory implementation
- Error handling and graceful degradation
- Modular and maintainable code structure

## Future Development Roadmap

### Planned Features
1. **Enhanced Citation System**
   - Implement detailed citation tracking
   - Add citation export functionality
   - Create citation networks visualization

2. **Advanced Analytics**
   - Add research trend analysis
   - Implement document clustering
   - Create visualization for knowledge graphs

3. **System Improvements**
   - Multi-model support (GPT-4, Claude, etc.)
   - Automated PDF metadata extraction
   - Enhanced conversation memory management
   - Chat export functionality

4. **User Experience**
   - Custom embedding model fine-tuning
   - Advanced search filters
   - User feedback integration
   - Collaborative features

## Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Required packages: `streamlit`, `langchain`, `chromadb`, `openai`

### Installation
```bash
git clone [repository-url]
cd biorag
pip install -r requirements.txt
```

### Configuration
1. Create a `.env` file in the project root
2. Add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key
```

### Running the Application
```bash
streamlit run app.py
```

## Live Demo
Try the live demo at: [BioRAG Oocyte Expert](https://biorag-oocyte-36nfepumrpgfwushlci6c2.streamlit.app/)

## Skills Demonstrated
- Large Language Models (LLM) Integration
- Natural Language Processing (NLP)
- Vector Embeddings & Semantic Search
- RAG System Architecture
- Python Development
- Web Application Development
- Scientific Document Processing
- Machine Learning Engineering
- Data Engineering
- API Integration
- Error Handling
- State Management
- User Interface Design

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
*This project was created as part of a portfolio demonstrating Machine Learning Engineering and Data Science capabilities, specifically focusing on NLP, RAG systems, and LLM integration.*