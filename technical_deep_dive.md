# BioRAG 系统技术深度拆解 — 逐文件、逐函数、逐行解读

> 这份文档的目的：让你在面试前彻底搞懂自己写的每一行代码。每个文件、每个类、每个函数都会讲清楚 **做了什么 → 为什么这么做 → 输入输出是什么 → 面试官会怎么问 → 你应该怎么回答**。

---

## 目录

1. [系统全景图 — 数据怎么从 PDF 变成回答](#1-系统全景图)
2. [文件 1: `src/document_loader.py` — 文档加载与分块](#2-document_loaderpy)
3. [文件 2: `src/embeddings.py` — 向量存储管理](#3-embeddingspy)
4. [文件 3: `src/rag_pipeline.py` — RAG 核心流水线](#4-rag_pipelinepy)
5. [文件 4: `process_pdfs.py` — 离线预处理脚本](#5-process_pdfspy)
6. [文件 5: `app.py` — Streamlit Web 应用](#6-apppy)
7. [文件 6: `requirements.txt` — 依赖清单](#7-requirementstxt)
8. [端到端数据流总结](#8-端到端数据流总结)

---

## 1. 系统全景图

先看大图，你的系统分两个阶段运行：

```
阶段一：离线预处理（只跑一次）
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────┐
│  PDF 论文    │ →  │ DocumentProcessor │ →  │ VectorStoreManager│ →  │  ChromaDB    │
│  (data/papers) │    │ (加载+分块)       │    │ (编码+存储)        │    │ (持久化向量库) │
└─────────────┘    └──────────────────┘    └─────────────────┘    └──────────────┘
     process_pdfs.py 驱动这个流程

阶段二：在线问答（用户每次提问）
┌─────────────┐    ┌──────────────┐    ┌──────────────────────────┐    ┌───────────┐
│  用户问题    │ →  │  ChromaDB    │ →  │ ConversationalRetrieval  │ →  │  回答+引用  │
│  (自然语言)  │    │ (检索top-4)  │    │ Chain (GPT-3.5-turbo)    │    │  (展示在UI) │
└─────────────┘    └──────────────┘    └──────────────────────────┘    └───────────┘
     app.py + rag_pipeline.py 驱动这个流程
```

面试时记住这个两阶段架构。面试官问"walk me through your system"，就按这个顺序讲。

---

## 2. `document_loader.py`

这是系统的**第一站**——把 PDF 文件变成 LLM 能处理的文本块。

### 完整源码（21 行）

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def load_pdfs(self, directory_path):
        """Load all PDFs from specified directory"""
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
```

### 逐块拆解

#### Import 部分

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

| 组件 | 作用 |
|------|------|
| `PyPDFLoader` | 读取单个 PDF 文件，提取每一页的纯文本 |
| `DirectoryLoader` | 扫描一个目录，批量加载所有匹配文件 |
| `RecursiveCharacterTextSplitter` | 把长文本切成小块，是 LangChain 最常用的分块器 |

#### `__init__` 方法

```python
def __init__(self):
    self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
```

**做了什么：** 创建一个文本分割器实例，配置好分块参数。

**三个参数的含义：**

| 参数 | 值 | 含义 |
|------|----|------|
| `chunk_size` | 1000 | 每个块最多 1000 个字符（大约 150-200 个英文单词，相当于一个自然段） |
| `chunk_overlap` | 200 | 相邻两个块之间有 200 字符的重叠 |
| `length_function` | `len` | 用 Python 内置的 `len()` 来计算文本长度（按字符数） |

**为什么要重叠（overlap）？** 举个例子：

假设原文是：
> "BMP15 is a key growth factor. It activates the SMAD signaling pathway, which regulates oocyte maturation."

如果在 "pathway" 之后刚好切断，没有重叠的话：
- 块 A: "...It activates the SMAD signaling pathway,"
- 块 B: "which regulates oocyte maturation."

块 B 孤立地看，你不知道"which"指的是什么。有了 200 字符的重叠：
- 块 A: "...It activates the SMAD signaling pathway, which regulates oocyte maturation."
- 块 B: "...SMAD signaling pathway, which regulates oocyte maturation. [后续内容]..."

这样 BMP15 → SMAD → oocyte maturation 的关系在至少一个块中是完整的。

**`RecursiveCharacterTextSplitter` 的"递归"是什么意思？** 它会按优先级尝试多种分隔符来切割：
1. 先尝试 `\n\n`（段落分隔）
2. 再尝试 `\n`（换行）
3. 再尝试 ` `（空格）
4. 最后逐字符切割

这样尽量在语义自然的地方断开，而不是在单词中间硬切。

#### `load_pdfs` 方法

```python
def load_pdfs(self, directory_path):
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return self.text_splitter.split_documents(documents)
```

**输入：**
| 参数 | 类型 | 例子 |
|------|------|------|
| `directory_path` | `str` | `"data/papers"` |

**内部流程：**

```
第1步: DirectoryLoader 扫描 data/papers/ 下所有 .pdf 文件
       ↓
第2步: 对每个 PDF 用 PyPDFLoader 逐页提取文本
       ↓ 得到 List[Document]，每个 Document = 一页 PDF
       ↓ Document 对象有两个属性：
       ↓   .page_content = "这一页的文本内容"
       ↓   .metadata = {"source": "data/papers/paper1.pdf", "page": 0}
       ↓
第3步: text_splitter.split_documents() 把每页文本切成 ~1000 字符的块
       ↓ metadata 会被继承，每个块都知道自己来自哪个文件、哪一页
```

**输出：**
| 返回值 | 类型 | 说明 |
|--------|------|------|
| 文档块列表 | `List[Document]` | 每个元素是一个 ~1000 字符的文本块，带有 source 和 page 元数据 |

**具体例子：** 假设你有 3 篇论文，每篇 10 页，每页约 3000 字符：
- `loader.load()` 返回 30 个 Document（3 篇 × 10 页）
- `split_documents()` 把每页切成约 3-4 个块（3000 ÷ 1000，考虑重叠）
- 最终返回约 90-120 个块

### 面试问答

---

**Q: Why did you choose RecursiveCharacterTextSplitter over other splitters?**

> "RecursiveCharacterTextSplitter is the most commonly recommended splitter in LangChain for general-purpose text, and here's why. It attempts to split at semantically meaningful boundaries — paragraph breaks first, then sentence breaks, then word breaks — rather than just cutting at a fixed character count. For scientific papers, this is particularly important because a key finding like 'BMP15 activates SMAD signaling' should ideally stay within a single chunk. The recursive approach maximizes the chance of that happening.
>
> Alternatives I considered: CharacterTextSplitter just splits on a single separator, which is less flexible. TokenTextSplitter splits by token count, which is useful when you need precise token budgets for the LLM, but character-based splitting was sufficient for my use case. If I were handling more structured documents, I might use something like MarkdownTextSplitter or a custom splitter that respects section headers like Abstract, Methods, Results."

---

**Q: Your chunk_size is 1000 characters. How did you arrive at that number?**

> "It's a trade-off between three factors. First, retrieval precision: smaller chunks mean each chunk is about one specific idea, so cosine similarity search is more precise. But too small — say 200 characters — and you lose context. A chunk that says 'it was upregulated' is useless without knowing what 'it' refers to. Second, LLM context window: each retrieved chunk consumes tokens. I retrieve 4 chunks, so that's roughly 4000 characters or about 1000 tokens — well within GPT-3.5-turbo's 16K token window, leaving plenty of room for conversation history and the generated answer. Third, semantic completeness: 1000 characters is roughly one paragraph in a scientific paper, which typically contains one complete idea or finding.
>
> I experimented with 500 and 1500. At 500, too many retrieval results were fragmentary. At 1500, irrelevant information was getting mixed in with relevant content. 1000 was the sweet spot for this corpus."

---

**Q: What happens if a PDF has images, tables, or equations?**

> "PyPDFLoader only extracts text content — it cannot parse images, table structures, or mathematical equations. This is a known limitation. For scientific papers, this means experimental data in tables and method details in figures are lost. To address this, I'd consider Unstructured.io for table extraction, a multimodal model like GPT-4V for figure description, or LlamaParse which is specifically designed for RAG-friendly document parsing. But for my MVP, text-only extraction was sufficient to demonstrate the core RAG workflow."

---

## 3. `embeddings.py`

这是系统的**第二站**——把文本块变成向量，存进数据库。

### 完整源码（35 行）

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

class VectorStoreManager:
    def __init__(self):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings()

    def create_vector_store(self, documents, persist_directory="data/chroma_db"):
        """Create or update vector store"""
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        return vector_store

    def load_vector_store(self, persist_directory="data/chroma_db"):
        """Load existing vector store"""
        if not os.path.exists(persist_directory):
            raise ValueError("Vector store not found!")

        return Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
```

### 逐块拆解

#### `__init__` 方法

```python
def __init__(self):
    load_dotenv()                                              # 1
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # 2
    self.embeddings = OpenAIEmbeddings()                       # 3
```

三行代码做了三件事：

| 行 | 做了什么 | 为什么 |
|----|---------|--------|
| 1 | 从 `.env` 文件加载环境变量 | `.env` 里存了 `OPENAI_API_KEY=sk-xxxxx`，这行把它读进内存 |
| 2 | 把 API Key 设进 `os.environ` | 确保 OpenAI SDK 能通过环境变量找到 Key |
| 3 | 创建 OpenAI Embedding 实例 | 默认使用 `text-embedding-ada-002` 模型，输出 1536 维向量 |

**关于 Embedding 模型的技术细节：**
- 模型名：`text-embedding-ada-002`
- 输出维度：1536
- 原理：把任意长度的文本映射到一个 1536 维的浮点数向量
- 性质：语义相似的文本 → 向量在空间中距离近（余弦相似度高）

举例：
```
"oocyte maturation process"  →  [0.012, -0.034, 0.056, ..., 0.078]  (1536个数)
"egg cell development"       →  [0.011, -0.031, 0.058, ..., 0.075]  (很接近!)
"weather forecast tomorrow"  →  [0.892, 0.445, -0.223, ..., -0.567] (很远!)
```

#### `create_vector_store` 方法

```python
def create_vector_store(self, documents, persist_directory="data/chroma_db"):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=self.embeddings,
        persist_directory=persist_directory
    )
    return vector_store
```

**输入：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `documents` | `List[Document]` | 上一步 `DocumentProcessor.load_pdfs()` 返回的文本块列表 |
| `persist_directory` | `str` | 向量库存储路径，默认 `"data/chroma_db"` |

**内部流程：**

```
第1步: 检查存储目录是否存在，不存在就创建
       ↓
第2步: Chroma.from_documents() 做了两件事：
       a) 对每个 Document 的 page_content 调用 OpenAI Embedding API
          "BMP15 activates SMAD..." → [0.012, -0.034, ..., 0.078] (1536维)
       b) 把向量 + 原文 + metadata 存进 ChromaDB
       （ChromaDB 0.4+ 在指定 persist_directory 后会自动持久化到磁盘，无需显式调用 persist()）
       生成文件在 data/chroma_db/ 目录下
```

**输出：**
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `vector_store` | `Chroma` | 可以直接用来做相似度搜索的向量库对象 |

**实际发生了什么？** 假设你有 100 个文本块：
- 向 OpenAI API 发送 100 次 Embedding 请求（或批量发送）
- 每个块变成 1536 个浮点数
- 100 个向量 + 100 段原文 + 100 条 metadata 存入 ChromaDB
- ChromaDB 0.4+ 在指定 `persist_directory` 后自动持久化到 `data/chroma_db/` 目录（SQLite + 索引文件），无需显式调用 `persist()`

#### `load_vector_store` 方法

```python
def load_vector_store(self, persist_directory="data/chroma_db"):
    if not os.path.exists(persist_directory):
        raise ValueError("Vector store not found!")

    return Chroma(
        embedding_function=self.embeddings,
        persist_directory=persist_directory
    )
```

**输入：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `persist_directory` | `str` | 之前创建的向量库路径 |

**做了什么：** 从磁盘加载已有的向量库。注意它不需要重新编码文档——向量已经存好了。它只需要 `embedding_function` 是因为后续查询时需要把用户的问题也编码成向量。

**输出：**
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `Chroma` 对象 | `Chroma` | 从磁盘恢复的向量库，可以直接搜索 |

**`create` vs `load` 的区别：**
- `create_vector_store`：第一次跑，从零开始编码文档并存储。慢（要调 API）。
- `load_vector_store`：后续每次启动应用时用，只是从磁盘读取已有数据。快（不调 API）。

### 面试问答

---

**Q: Why ChromaDB and not Pinecone, FAISS, or Weaviate?**

> "I chose ChromaDB for three reasons specific to this project's constraints. First, local persistence: ChromaDB stores data as files on disk — no need for a running server process or cloud service. For a prototype with three papers, this is ideal. Second, LangChain integration: ChromaDB has first-class support in LangChain — `Chroma.from_documents()` handles embedding and indexing in a single call. Third, zero infrastructure: no Docker, no API keys for the database, no cost.
>
> The trade-offs are clear. ChromaDB doesn't support horizontal scaling — if I had millions of vectors, I'd need Pinecone for managed scaling or Milvus for self-hosted distributed search. FAISS, from Facebook, would give me better raw search performance, but it doesn't have built-in persistence — I'd need to manage save/load logic myself. For this project's scope — thousands of vectors, single-user — ChromaDB was the right call."

---

**Q: What embedding model are you using, and what are its characteristics?**

> "I'm using OpenAI's text-embedding-ada-002, which outputs 1536-dimensional dense vectors. It's trained on a large corpus with a contrastive learning objective, meaning it learns to place semantically similar texts close together in vector space and dissimilar texts far apart. It supports up to 8191 tokens of input.
>
> Key characteristics: it's multilingual, so a query in English can match a passage about the same concept in Chinese. It produces normalized vectors, which means cosine similarity and dot product give the same ranking. And it's relatively cheap — about $0.0001 per 1000 tokens.
>
> If I were to improve this, I'd consider domain-specific embedding models like BiomedBERT or PubMedBERT embeddings, which are fine-tuned on biomedical literature and might better capture specialized terminology like gene names and pathway terms."

---

**Q: Why don't you call `persist()` after creating the vector store?**

> "Starting with ChromaDB 0.4+, explicit `persist()` calls are no longer needed — and in fact are deprecated. When you pass a `persist_directory` to `Chroma.from_documents()`, ChromaDB automatically persists data to disk on every write operation. Under the hood, it uses SQLite for metadata storage and a custom index format for the vectors. The directory contains files like `chroma.sqlite3` for metadata and index files for the vector index. This means if the process restarts, we can reload the exact same state without re-computing embeddings — which saves both time and API costs. The older pattern of calling `vector_store.persist()` manually was required in ChromaDB 0.3.x but is now unnecessary and will emit a deprecation warning."

---

## 4. `rag_pipeline.py`

这是系统的**大脑**——把检索和生成串在一起，实现对话式问答。

### 完整源码

```python
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

class RAGPipeline:
    def __init__(self, vector_store):
        # 接收外部传入的向量库实例，避免重复创建
        self.vector_store = vector_store

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0),
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            return_source_documents=True,
            output_key="answer",
            verbose=True
        )

    def ask(self, query: str):
        response = self.qa_chain({"question": query})
        return response

if __name__ == '__main__':
    from src.embeddings import VectorStoreManager
    vector_store_manager = VectorStoreManager()
    vector_store = vector_store_manager.load_vector_store()
    rag = RAGPipeline(vector_store)
    while True:
        user_input = input("Please type your question:")
        if user_input.lower() == 'exit':
            break
        result = rag.ask(user_input)
        print("Answer：", result['answer'])
```

### 逐块拆解

#### `__init__` 方法 — 三个核心组件的组装

这个 `__init__` 组装了整个 RAG 流水线的三个关键零件：

**零件 1: 向量库（Retriever）**
```python
self.vector_store = vector_store
```
接收外部传入的向量库实例（由 `VectorStoreManager.load_vector_store()` 创建）。这样避免了重复创建 Chroma 连接和 Embedding 模型，实现了职责分离：`VectorStoreManager` 负责管理向量库的创建和加载，`RAGPipeline` 只负责问答。

**零件 2: 对话记忆**
```python
self.memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)
```

| 参数 | 值 | 含义 |
|------|----|------|
| `memory_key` | `"chat_history"` | 记忆存储在字典的 `chat_history` 键下 |
| `return_messages` | `True` | 以 Message 对象列表的格式返回历史（而不是纯字符串拼接） |
| `output_key` | `"answer"` | 指定从 chain 输出中取哪个 key 作为 AI 回复存入记忆。因为 chain 返回多个 key（answer、source_documents），不指定会报错 |

**ConversationBufferMemory 的工作方式：**

```
第1轮:
  用户: "What is BMP15?"
  AI: "BMP15 is a growth factor..."
  → memory 存储: [HumanMessage("What is BMP15?"), AIMessage("BMP15 is a growth factor...")]

第2轮:
  用户: "How does it relate to oocyte maturation?"
  → memory 把之前的记录一起发给 LLM，这样 LLM 知道 "it" = BMP15

第3轮:
  → memory 越来越长... 这就是 BufferMemory 的问题
```

**零件 3: 对话检索链（核心！）**
```python
self.qa_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0),                                      # LLM
    retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}), # 检索器
    memory=self.memory,                                              # 记忆
    return_source_documents=True,                                     # 返回引用
    output_key="answer",                                              # 输出键名
    verbose=True                                                      # 打印调试日志
)
```

**每个参数的作用：**

| 参数 | 值 | 作用 |
|------|----|------|
| `ChatOpenAI(temperature=0)` | GPT-3.5-turbo | 负责生成回答。temperature=0 → 确定性输出，不要创造性 |
| `retriever` | ChromaDB retriever | 负责从向量库检索相关文档。`k=4` 表示返回最相似的 4 个块 |
| `memory` | ConversationBufferMemory | 维护对话历史，支持多轮问答 |
| `return_source_documents` | `True` | 在响应中附带检索到的原始文档（用于展示引用） |
| `output_key` | `"answer"` | 指定 chain 输出中回答文本的键名，与 memory 的 `output_key` 对应，确保记忆正确记录 AI 的回复 |
| `verbose` | `True` | 在终端打印 Chain 的执行日志（调试用） |

**`as_retriever(search_kwargs={"k": 4})` 做了什么？**
把 ChromaDB 向量库包装成一个 LangChain `Retriever` 对象。当被调用时：
1. 接收查询文本
2. 用 `self.embeddings` 把查询编码成向量
3. 在 ChromaDB 中做余弦相似度搜索
4. 返回 top-4 最相似的 Document 对象

**ConversationalRetrievalChain 内部的完整执行流程（重要！面试必考）：**

```
用户输入: "Are any of those druggable?"
            ↓
┌─────────────────────────────────────────────┐
│  Step 1: Question Condensing（问题重写）       │
│                                              │
│  输入: 当前问题 + chat_history                 │
│    "Are any of those druggable?"             │
│    + [之前问了 "What pathways regulate        │
│       oocyte maturation?"]                    │
│                                              │
│  → LLM 重写为独立问题:                         │
│    "Are any pathways that regulate oocyte     │
│     maturation associated with druggable      │
│     targets?"                                 │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│  Step 2: Retrieval（向量检索）                 │
│                                              │
│  用重写后的独立问题去 ChromaDB 搜索              │
│  → 返回 top-4 最相似的文本块                    │
│  每个块带有 page_content 和 metadata           │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│  Step 3: Generation（答案生成）                │
│                                              │
│  把 4 个文本块 + 重写后的问题 一起发给 GPT-3.5  │
│  Prompt 大致是:                                │
│    "Based on the following context, answer    │
│     the question.                             │
│     Context: [4个文本块]                       │
│     Question: [重写后的问题]"                   │
│                                              │
│  → GPT-3.5 基于这些上下文生成回答               │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│  Step 4: Memory Update（记忆更新）             │
│                                              │
│  把这轮的问答对追加到 chat_history               │
│  下次提问时会带上完整的对话历史                   │
└─────────────────────────────────────────────┘
```

**为什么 Step 1（问题重写）是关键？**

没有问题重写的话：
- 用户问："Are any of those druggable?"
- "those" 对向量库来说毫无意义
- 检索结果会完全不相关
- 回答质量崩溃

有问题重写后：
- LLM 把 "those" 解析为之前提到的 pathways
- 生成独立问题："Are any pathways regulating oocyte maturation druggable?"
- 向量库检索到正确的文档
- 回答质量保持高水平

#### `ask` 方法

```python
def ask(self, query: str):
    response = self.qa_chain({"question": query})
    return response
```

**输入：**
| 参数 | 类型 | 例子 |
|------|------|------|
| `query` | `str` | `"What signaling pathways regulate oocyte maturation?"` |

**输出：**
| 返回值 | 类型 | 结构 |
|--------|------|------|
| `response` | `dict` | 包含三个键（见下表） |

```python
{
    "question": "What signaling pathways regulate oocyte maturation?",
    "answer": "Based on the literature, several signaling pathways regulate oocyte maturation, including the MAPK/ERK pathway, PI3K/AKT pathway, and BMP/SMAD pathway...",
    "source_documents": [
        Document(page_content="...", metadata={"source": "paper1.pdf", "page": 3}),
        Document(page_content="...", metadata={"source": "paper2.pdf", "page": 7}),
        Document(page_content="...", metadata={"source": "paper1.pdf", "page": 5}),
        Document(page_content="...", metadata={"source": "paper3.pdf", "page": 2})
    ]
}
```

#### `__main__` 部分

```python
if __name__ == '__main__':
    from src.embeddings import VectorStoreManager
    vector_store_manager = VectorStoreManager()
    vector_store = vector_store_manager.load_vector_store()
    rag = RAGPipeline(vector_store)
    while True:
        user_input = input("Please type your question:")
        if user_input.lower() == 'exit':
            break
        result = rag.ask(user_input)
        print("Answer：", result['answer'])
```

这是一个命令行测试接口。直接运行 `python src/rag_pipeline.py` 就可以在终端里和系统对话，不需要启动 Streamlit。方便调试。注意这里通过 `VectorStoreManager` 加载向量库再传给 `RAGPipeline`，保持了和 `app.py` 一致的初始化方式。

### 面试问答

---

**Q: Walk me through what happens internally when a user asks a question.**

> "When the user submits a query, the ConversationalRetrievalChain executes a three-step pipeline. First, question condensing: if there's any conversation history, the chain sends the current question along with the chat history to the LLM and asks it to rewrite the question as a standalone query. For example, 'How does it affect fertility?' becomes 'How does BMP15 affect oocyte fertility?' This is critical because the vector database has no concept of conversational context.
>
> Second, retrieval: the condensed question is embedded using OpenAI's embedding model into a 1536-dimensional vector, then ChromaDB performs a cosine similarity search and returns the top-4 most similar document chunks. Each chunk carries metadata including the source file and page number.
>
> Third, generation: the 4 retrieved chunks are injected into a prompt template as context, along with the condensed question. GPT-3.5-turbo generates an answer grounded in that context. Finally, the memory module stores this Q&A pair for future turns."

---

**Q: What's the problem with ConversationBufferMemory? How would you fix it?**

> "ConversationBufferMemory stores the entire conversation history verbatim. Every human message and AI response is appended to a growing list. After 10-15 exchanges, this can consume several thousand tokens, eating into GPT-3.5-turbo's 16K context window and leaving less room for retrieved documents and the actual answer.
>
> I'd fix this in stages. The quickest fix is ConversationBufferWindowMemory with k=5 — keep only the last 5 exchanges, drop everything older. A more sophisticated approach is ConversationSummaryBufferMemory — it keeps recent messages verbatim but summarizes older ones using the LLM, compressing 'we discussed BMP15, SMAD signaling, and oocyte maturation pathways' into a single sentence. The best production approach is a hybrid: recent 3 turns kept in full, everything older summarized, with a hard token cap."

---

**Q: Why `temperature=0`? Would you ever change it?**

> "Temperature controls the probability distribution over the next token during generation. At zero, the model always picks the most probable token — effectively greedy decoding. This makes output deterministic and maximally factual, which is essential for scientific Q&A. A researcher asking about cell signaling pathways needs a reproducible, evidence-based answer, not creative prose.
>
> I'd increase temperature in specific scenarios: hypothesis generation, where you want the model to suggest non-obvious connections, maybe 0.3 to 0.5. Or diverse summarization, where you want multiple distinct phrasings of the same concept. But for any customer-facing QIAGEN product serving pharma researchers, I'd default to low temperature. Reproducibility is a requirement, not a nice-to-have."

---

**Q: What does `return_source_documents=True` give you?**

> "It instructs the chain to include the actual Document objects that were retrieved from ChromaDB in the response dictionary. Each Document has two attributes: `page_content`, which is the raw text of that chunk, and `metadata`, which contains the source PDF filename and page number. This is what powers the citation feature in the UI — I can show users not just the answer, but exactly which passages from which papers the answer is based on. In a pharma context, this traceability is non-negotiable. A scientist needs to verify any AI-generated claim against the primary source."

---

## 5. `process_pdfs.py`

这是一个**一次性运行的脚本**——执行阶段一（离线预处理），把 PDF 变成向量库。

### 完整源码（49 行）

```python
from src.document_loader import DocumentProcessor
from src.embeddings import VectorStoreManager
import os

def main():
    pdf_directory = "data/papers"

    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        print(f"Created directory {pdf_directory}")
        print(f"Please place your PDF files in {pdf_directory} directory and run this script again.")
        return

    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}!")
        print("Please add some PDF files to this directory and run this script again.")
        return

    print(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")

    print("Processing PDF documents...")
    document_processor = DocumentProcessor()
    documents = document_processor.load_pdfs(pdf_directory)

    if not documents:
        print("Error: No document chunks were generated!")
        return

    print(f"Successfully processed {len(documents)} document chunks.")

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
```

### 逐块拆解

这个文件是一个**编排脚本**——它本身不包含新逻辑，而是按正确的顺序调用前面两个模块。

**完整执行流程：**

```
python process_pdfs.py
        ↓
检查 data/papers/ 目录是否存在 → 不存在就创建并提示用户放入 PDF
        ↓
检查目录里有没有 .pdf 文件 → 没有就提示用户添加
        ↓
DocumentProcessor().load_pdfs("data/papers")
  → 读取所有 PDF → 提取文本 → 切分成 ~1000 字符的块
  → 返回 List[Document]
        ↓
打印 "Successfully processed X document chunks."
        ↓
VectorStoreManager().create_vector_store(documents)
  → 对每个块调用 OpenAI Embedding API → 存入 ChromaDB → 持久化到磁盘
        ↓
打印 "Vector store created successfully!"
        ↓
现在可以运行 streamlit run app.py 了
```

**为什么要把预处理独立成一个脚本？**

三个原因：
1. **Embedding 很贵也很慢：** 100 个文本块需要约 100 次 API 调用。如果每次启动 app 都重新编码，浪费时间和钱。
2. **数据不常变：** 论文库不是每天都更新的。预处理一次，向量库存到磁盘，app 启动时直接加载。
3. **关注点分离：** 数据准备和用户交互是两个不同的关注点。分开后，你可以单独更新数据（加新论文 → 重跑 `process_pdfs.py`），而不影响 app 的代码。

**防御性编程细节：**
- 检查目录是否存在
- 检查是否有 PDF 文件
- 检查分块结果是否为空
- 捕获向量存储创建过程中的异常（通常是 API Key 问题）

### 面试问答

---

**Q: Why did you separate the PDF processing from the main app?**

> "It's a classic separation of offline and online workloads. Embedding documents is an expensive, one-time operation — it calls the OpenAI API for every chunk and can take minutes. The web application, on the other hand, needs to start quickly and serve user queries in real time. By processing PDFs offline and persisting the vector store to disk, the app startup just loads pre-computed vectors from disk — instant. This also saves API costs: if I restart the app 10 times during development, I'm not re-embedding the same documents 10 times. In a production system, this separation would be even more important — you'd likely run the ingestion pipeline on a schedule or as a triggered job, completely decoupled from the serving layer."

---

## 6. `app.py`

这是**用户看到的界面**——Streamlit Web 应用，把所有后端模块串成一个可交互的产品。

### 完整源码（约 200 行）— 分四个逻辑段落讲解

#### 段落 1: 配置与样式（第 1-49 行）

```python
import streamlit as st
from src.embeddings import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import os

load_dotenv()

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

st.markdown("""<style>...</style>""", unsafe_allow_html=True)
```

**做了什么：**
1. **环境变量加载：** `load_dotenv()` 从 `.env` 文件加载环境变量（如 `OPENAI_API_KEY`），支持本地开发。
2. **双重 API Key 获取：** 先尝试从环境变量（`.env`）获取 API Key；如果不存在，尝试从 Streamlit Cloud Secrets 获取。这样同时支持本地开发和 Streamlit Cloud 部署两种场景。如果两者都没有，显示错误并停止。
3. **页面配置：** `st.set_page_config()` 设置浏览器标签标题、图标、页面布局。`layout="wide"` 让页面使用全宽而不是默认的居中窄列。
4. **自定义 CSS：** 通过 `st.markdown` 注入 CSS 来美化聊天界面。定义了 `.chat-message`、`.user-message`、`.assistant-message` 和 `.citation` 四个样式类。

#### 段落 2: Session State 初始化（第 52-57 行）

```python
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False
```

**为什么需要 session_state？**

Streamlit 的执行模型很特殊：**每次用户交互（点击按钮、输入文字），整个 `app.py` 脚本会从头到尾重新执行一遍。** 普通的 Python 变量每次重新执行都会被重置。`st.session_state` 是 Streamlit 提供的持久化存储——跨次运行保留数据。

| 状态变量 | 类型 | 用途 |
|----------|------|------|
| `chat_history` | `List[dict]` | 存储所有聊天消息（用户+AI），用于渲染聊天界面 |
| `rag_pipeline` | `RAGPipeline` or `None` | RAG 流水线实例（内部包含向量库），避免每次交互都重新创建 |
| `is_initialized` | `bool` | 标记系统是否已初始化，避免重复加载 |

**举例说明 Streamlit 的重新执行机制：**

```
用户第1次打开页面:
  app.py 执行 → session_state 为空 → 初始化所有变量 → 加载向量库 → 渲染空聊天界面

用户输入 "What is BMP15?" 并按回车:
  app.py 从头执行 → session_state 已有数据 → 跳过初始化 → 渲染之前的聊天 → 处理新问题

用户再输入 "How does it work?":
  app.py 从头执行 → session_state 有 2 条消息 → 渲染之前的聊天 → 处理新问题
```

如果不用 `session_state`，每次用户输入后，对话历史和 RAG 流水线都会被清空，多轮对话就不可能实现。

#### 段落 3: 侧边栏和系统初始化（第 60-108 行）

```python
# 侧边栏
with st.sidebar:
    st.title("🧬 Oocyte Expert")
    st.markdown("""...""")

    if st.session_state.is_initialized:
        st.success("Knowledge Base: Active ✅")
    else:
        st.warning("Knowledge Base: Loading...")

    if st.button("Reset System"):
        st.session_state.chat_history = []
        st.rerun()

# 系统初始化（只在首次运行时执行）
if not st.session_state.is_initialized:
    with st.spinner("Initializing knowledge base..."):
        try:
            vector_store_manager = VectorStoreManager()
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
```

**侧边栏做了什么：**
- 展示项目标题和简介
- 用 `st.success` / `st.warning` 显示系统状态（绿色 = 就绪，黄色 = 加载中）
- "Reset System" 按钮清空对话历史并刷新页面

**初始化流程：**

```
is_initialized == False? (首次访问)
        ↓ Yes
显示 spinner "Initializing knowledge base..."
        ↓
VectorStoreManager() → 初始化 Embedding 模型
        ↓
load_vector_store("data/chroma_db") → 尝试从磁盘加载向量库
        ↓ 如果向量库不存在 → 自动构建向量库：
        ↓   1. 显示 st.info 提示用户正在构建
        ↓   2. DocumentProcessor().load_pdfs("data/papers") 加载并分块
        ↓   3. create_vector_store(documents) 编码并持久化
        ↓
RAGPipeline(vector_store) → 复用已加载的向量库，创建 RAG 流水线
        ↓
is_initialized = True → 下次脚本重新执行时跳过这个块
```

**`st.stop()` 的作用：** 立即停止脚本执行，页面只显示到目前为止渲染的内容。这是一种优雅的错误处理——如果在自动构建过程中找不到 PDF 文件，不要继续渲染聊天界面。

#### 段落 3.5: 推荐问题（Suggested Questions）

```python
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

if not st.session_state.chat_history:
    st.markdown("### Try asking:")
    for category, questions in SUGGESTED_QUESTIONS.items():
        st.markdown(f"**{category}**")
        for q in questions:
            if st.button(q, key=q):
                st.session_state.pending_question = q
                st.rerun()
```

**做了什么：**
- 定义了一组按类别分组的推荐问题，帮助用户快速开始对话
- 只在聊天历史为空时显示（一旦用户开始对话，推荐问题消失）
- 用户点击某个推荐问题后，把它存入 `st.session_state.pending_question`，然后调用 `st.rerun()` 重新执行脚本
- 在下一次执行中，聊天输入处理逻辑会读取 `pending_question` 并作为用户的提问处理

#### 段落 4: 聊天界面

```python
# 渲染历史消息
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

# 底部按钮
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()
with col2:
    if st.button("Export Chat"):
        st.info("Export feature coming soon!")
```

**聊天渲染流程：**

```
第1步: 遍历 chat_history，渲染之前所有的对话消息
       每条消息用 st.chat_message() 显示对应的头像（user/assistant）
       如果消息有 citations，用 st.expander 折叠展示
       ↓
第2步: 检查是否有 pending_question（来自推荐问题点击）
       以及 st.chat_input() 显示输入框，等待用户输入
       prompt = pending or typed
       ↓ 用户输入 "What is BMP15?" 并按回车（或点击推荐问题）
       ↓
第3步: 把用户消息追加到 chat_history
       ↓
第4步: 调用 rag_pipeline.ask(prompt)
       → Question Condensing → Retrieval → Generation
       ↓
第5步: 从 response 中提取 answer 和 source_documents
       遍历 source_documents，提取每个文档的 source 文件名和 page 页码
       生成引用列表（如 "paper1.pdf, Page 5"）
       ↓
第6步: 把 AI 回复追加到 chat_history（带真实 citations）
       ↓
第7步: st.write(answer) 显示回答
       ↓
第8步: 脚本执行完毕，页面更新，显示完整对话
```

**输入处理：`pending` vs `typed`**

```python
pending = st.session_state.pop("pending_question", None)
typed = st.chat_input("Ask your question about oocyte research...")
prompt = pending or typed
```

这里用两步获取用户输入：
- `pending`：来自推荐问题的点击（上一轮 rerun 时存入 `session_state.pending_question`）。`pop()` 取出后立即删除，避免重复处理。
- `typed`：用户在输入框中手动输入的文本。
- `prompt = pending or typed`：优先处理推荐问题点击，否则处理手动输入。两者都为 `None` 时不执行。

**底部按钮：**
- **Clear Conversation：** 清空 `chat_history` 并调用 `st.rerun()` 强制重新执行脚本，这样界面立刻更新。
- **Export Chat：** 占位功能，目前只显示 "coming soon" 提示。

### 面试问答

---

**Q: Explain how Streamlit's execution model works and how it affects your architecture.**

> "Streamlit has a unique execution model: every time a user interacts with the app — clicks a button, types in the chat input, toggles a checkbox — the entire Python script re-executes from top to bottom. This means any regular Python variable is reset on every interaction. That's why `st.session_state` is essential — it's a persistent key-value store that survives across re-executions within the same browser session.
>
> This model has architectural implications. Heavy initialization — like loading the vector store and creating the RAG pipeline — must be gated behind an `is_initialized` flag in session_state, otherwise it would re-run on every keystroke. The chat history must also live in session_state so previous messages aren't lost. And the RAG pipeline itself — including its conversation memory — must be stored in session_state so multi-turn context is preserved.
>
> The benefit of this model is simplicity: the script reads linearly, top to bottom, like a page layout. The trade-off is that you have to be explicit about what state persists and what doesn't."

---

**Q: How does the citation feature work?**

> "The citation feature extracts real source information from the retrieved documents. When `RAGPipeline.ask()` returns a response, it includes a `source_documents` list where each document carries `metadata` with the source PDF filename and page number. The app iterates over these documents, extracts the `source` and `page` fields from each document's metadata, and formats them as readable citations like 'paper1.pdf, Page 5'. The page number is incremented by 1 since PyPDFLoader uses zero-based page indexing. These citations are stored alongside the assistant's response in `chat_history` and displayed in an expandable `st.expander('View Citations')` panel beneath each answer. This gives researchers full traceability — they can verify any AI-generated claim against the specific passage in the original paper."

---

**Q: What happens if the vector store doesn't exist when the app starts?**

> "The app handles this gracefully through an auto-build fallback. In the initialization block, it first tries to call `load_vector_store('data/chroma_db')`. If that directory doesn't exist, the method raises a `ValueError`. Instead of stopping, the app catches this exception and automatically builds the vector store on the fly: it imports `DocumentProcessor`, loads and splits all PDFs from `data/papers/`, and calls `create_vector_store()` to embed and persist them. The user sees an informational message — 'Building vector store for the first time, this may take a moment...' — and the app continues normally once the build completes. This eliminates the need to run `process_pdfs.py` separately before launching the app. The only hard failure is if no PDF documents are found in `data/papers/`, in which case it displays an error and stops."

---

## 7. `requirements.txt`

```
streamlit>=1.40.0
langchain>=0.3.0
langchain-community>=0.3.0
langchain-openai>=0.2.0
openai>=1.60.0
chromadb>=0.5.0
python-dotenv>=1.0.0
pypdf>=4.0.0
tiktoken>=0.7.0
```

每个依赖的作用：

| 包 | 版本 | 用在哪里 | 做什么 |
|----|------|---------|--------|
| `streamlit` | >=1.40.0 | `app.py` | Web UI 框架，提供 chat 组件、session state、部署 |
| `langchain` | >=0.3.0 | 全部 `src/` | LLM 应用编排框架，提供核心抽象和编排逻辑 |
| `langchain-community` | >=0.3.0 | `src/` | LangChain 社区集成包，提供 PyPDFLoader、Chroma 等 |
| `langchain-openai` | >=0.2.0 | `src/` | LangChain 的 OpenAI 专用集成，提供 ChatOpenAI 和 OpenAIEmbeddings |
| `openai` | >=1.60.0 | (间接) | OpenAI Python SDK，langchain-openai 底层依赖 |
| `chromadb` | >=0.5.0 | `src/embeddings.py` | 向量数据库，存储和检索文档嵌入（0.4+ 自动持久化） |
| `python-dotenv` | >=1.0.0 | `src/embeddings.py`, `app.py` | 从 `.env` 文件加载环境变量（API Key） |
| `pypdf` | >=4.0.0 | (间接) | PyPDFLoader 底层依赖，负责 PDF 文件解析（替代已弃用的 pypdf2） |
| `tiktoken` | >=0.7.0 | (间接) | OpenAI 的 tokenizer，LangChain 用它来计算 token 数 |

### 面试问答

---

**Q: Why do you use `>=` version ranges instead of exact pins in requirements.txt?**

> "I switched from exact version pinning (`==`) to minimum version constraints (`>=`) for pragmatic reasons. With exact pins, every security patch or bug fix requires manually updating the version. Using `>=` with a tested minimum version — like `langchain>=0.3.0` — allows automatic adoption of compatible updates while guaranteeing the minimum feature set I depend on. LangChain's ecosystem has stabilized significantly since 0.3.x, with clearer separation between `langchain`, `langchain-community`, and `langchain-openai`, so the risk of breaking changes within a major version is lower. In a production environment, I'd complement this with a lock file — like `pip-compile` or Poetry's `poetry.lock` — to pin exact resolved versions for fully reproducible builds, while keeping `requirements.txt` as the looser specification."

---

## 8. 端到端数据流总结

把所有文件串起来，看一个完整的用户旅程：

### 旅程 1: 管理员准备知识库（跑一次，或由 app 自动构建）

```
管理员把 3 篇论文放进 data/papers/
         ↓
方式 A: 运行 python process_pdfs.py（离线预处理）
方式 B: 直接启动 app.py，如果向量库不存在会自动构建
         ↓
process_pdfs.py 或 app.py 自动构建流程:
  │
  ├─ DocumentProcessor.__init__()
  │    └─ 创建 TextSplitter(chunk_size=1000, overlap=200)
  │
  ├─ DocumentProcessor.load_pdfs("data/papers")
  │    ├─ DirectoryLoader 找到 3 个 PDF
  │    ├─ PyPDFLoader 提取每页文本 → 约 30 个 Document
  │    └─ TextSplitter 切分 → 约 100 个 Document 块
  │
  ├─ VectorStoreManager.__init__()
  │    ├─ load_dotenv() 读取 .env 文件
  │    └─ OpenAIEmbeddings() 初始化 embedding 模型
  │
  └─ VectorStoreManager.create_vector_store(100个块)
       ├─ Chroma.from_documents():
       │    ├─ 对 100 个块逐个调用 OpenAI Embedding API
       │    │    每个块 → 1536 维向量
       │    └─ 存入 ChromaDB 并自动持久化到 data/chroma_db/（SQLite + 索引文件）
       └─ 打印 "Vector store created successfully!"
```

### 旅程 2: 用户提问（每次交互）

```
用户打开浏览器访问 Streamlit 应用
         ↓
app.py 首次执行:
  ├─ load_dotenv() 加载 .env 环境变量
  ├─ 检查 OPENAI_API_KEY（先 .env，再 Streamlit Cloud Secrets）
  ├─ session_state 初始化（空列表、None、False）
  ├─ VectorStoreManager().load_vector_store("data/chroma_db")
  │    └─ 从磁盘加载向量库（不调 API，很快）
  │    └─ 如果不存在 → 自动构建：加载 data/papers/ 中的 PDF → 编码 → 持久化
  ├─ RAGPipeline(vector_store) — 复用已加载的向量库实例
  │    ├─ self.vector_store = vector_store — 直接使用传入的实例
  │    ├─ ConversationBufferMemory(output_key="answer") — 空的对话记忆
  │    └─ ConversationalRetrievalChain(output_key="answer") — 组装完整链
  └─ is_initialized = True
         ↓
显示推荐问题（SUGGESTED_QUESTIONS），或用户直接输入
用户输入: "What pathways regulate oocyte maturation?"
         ↓
app.py 重新执行:
  ├─ is_initialized == True → 跳过初始化
  ├─ 渲染空的聊天界面
  ├─ pending_question 或 st.chat_input 捕获用户输入
  ├─ 追加用户消息到 chat_history
  └─ rag_pipeline.ask("What pathways regulate oocyte maturation?")
       │
       ├─ ConversationalRetrievalChain 执行:
       │    │
       │    ├─ Step 1: Question Condensing
       │    │    chat_history 为空 → 直接用原问题
       │    │
       │    ├─ Step 2: Retrieval
       │    │    ├─ OpenAI Embedding API: 问题 → 1536 维向量
       │    │    ├─ ChromaDB: 余弦相似度搜索 top-4
       │    │    └─ 返回 4 个最相关的文档块
       │    │
       │    ├─ Step 3: Generation
       │    │    ├─ Prompt = "Based on context: [4个块], answer: [问题]"
       │    │    └─ GPT-3.5-turbo 生成回答
       │    │
       │    └─ Step 4: Memory Update
       │         └─ 存储 Q&A 对到 chat_history（通过 output_key="answer"）
       │
       └─ 返回 {"answer": "Several pathways...", "source_documents": [...]}
         ↓
提取 answer 和 source_documents → 构建 citations 列表
st.write(answer) → 在界面显示回答
追加 AI 消息到 chat_history（带真实引用信息）
         ↓
用户接着问: "Are any of those druggable?"
         ↓
app.py 重新执行:
  ├─ 渲染之前的 2 条消息（含可展开的引用）
  └─ rag_pipeline.ask("Are any of those druggable?")
       │
       └─ Step 1: Question Condensing
            ├─ chat_history = [之前的 Q&A]
            ├─ LLM 重写: "Are any pathways that regulate oocyte maturation druggable?"
            └─ 用重写后的问题去检索 → 生成 → 返回回答
```

### 面试终极问答

---

**Q: If I gave you 10 minutes to demo this system to a pharma customer, how would you structure it?**

> "I'd structure the 10 minutes into three acts. Act one — the problem, 2 minutes: 'Your team published three papers on oocyte biology. You want to ask questions across all of them simultaneously, with citations. Today that requires reading each paper end to end.' Act two — the solution, 6 minutes: live demo. I'd ask a domain-specific question like 'What signaling pathways regulate oocyte maturation?' and show the system returning an evidence-based answer with source citations. Then I'd ask a follow-up — 'Are any of those pathways druggable?' — to demonstrate multi-turn context. I'd point out that the system remembered what 'those' refers to. Act three — the bridge, 2 minutes: 'This is what I built with three papers and vector search. Imagine this at the scale of QIAGEN's Biomedical Knowledge Base — millions of curated findings, explicit pathway relationships, and graph-based reasoning. That's the step function improvement we can offer your team.'"

---

**Q: What's the single biggest weakness of this system?**

> "The system retrieves text chunks based on semantic similarity but has no understanding of biological entity relationships. If I ask 'What is upstream of SMAD signaling in oocyte maturation?', the system can only find chunks that happen to mention both 'upstream' and 'SMAD' — it doesn't actually traverse a signaling pathway. A knowledge graph would let me do exactly that: start at the SMAD node, follow 'activated_by' edges, and return the upstream regulators. That's the fundamental difference between what BioRAG does with vector search and what QIAGEN does with a curated biomedical knowledge graph. My project is a proof of concept for one half of the equation. QIAGEN has the other half — and the combination of both is where the real value lies."

---

> **复习建议：** 面试前把这份文档读两遍。第一遍读解释，理解每个组件的作用。第二遍只读面试问答部分，练习用英文回答。重点记住 ConversationalRetrievalChain 的三步内部流程（question condensing → retrieval → generation），这是最高频的技术面试题。
