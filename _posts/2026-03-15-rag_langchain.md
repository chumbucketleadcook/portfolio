---
layout: post
title: "Building a RAG Application with LangChain: Query Your Own Knowledge Base"
date: 2026-03-15
category: ml
---
# Building a RAG Application with LangChain: Query Your Own Knowledge Base

## Overview

Large language models have a fundamental limitation: their knowledge is frozen at training time. Ask a model about something that happened after its cutoff, or about a private document it's never seen, and it will either hallucinate or tell you it doesn't know.

Retrieval-Augmented Generation (RAG) solves this by splitting the problem in two. A retrieval system finds the relevant documents. A language model reads them and generates a response. The model doesn't need to have memorized your data — it just needs to be able to read and reason.

This post walks through building a RAG application from scratch using LangChain — specifically one that lets you query a collection of your own blog posts or technical documents. The architecture is straightforward enough to understand fully and extensible enough to serve as a template for production applications.

---

## Architecture Overview

The system has two phases that are worth keeping mentally separate:

**Indexing (offline)**: Documents are loaded, split into chunks, converted into vector embeddings, and stored in a vector database. This runs once (and incrementally when new documents are added).

**Querying (online)**: A user's question is embedded, similar chunks are retrieved from the vector database, and those chunks are passed alongside the question to a language model that generates the final answer.

```
INDEXING PHASE
Documents → Text Splitter → Embedding Model → Vector Store

QUERYING PHASE  
Question → Embedding Model → Vector Store (similarity search) → Top-K Chunks
                                                                      ↓
Question + Chunks → LLM → Answer
```

The key insight is that the LLM never needs to read your entire document corpus — it only reads the 3–5 chunks most relevant to the specific question being asked.

---

## Setup and Dependencies

```python
# requirements.txt additions
# langchain==0.3.26
# langchain-community==0.0.37
# langchain-anthropic (or langchain-openai)
# chromadb
# sentence-transformers
# pypdf  # if ingesting PDFs
```

For embeddings, we'll use `sentence-transformers` with the `all-MiniLM-L6-v2` model — it's small, fast, runs locally without an API key, and produces high-quality embeddings for semantic search. For the generative model, we'll use the Anthropic API (you can substitute OpenAI or your own SageMaker endpoint).

---

## Step 1: Document Loading

LangChain provides document loaders for virtually every source format. For a blog-based knowledge base, loading from a directory of markdown files is the natural starting point:

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from pathlib import Path

def load_blog_posts(posts_dir: str) -> list[Document]:
    """
    Load all markdown files from a Jekyll _posts directory.
    Returns a list of LangChain Document objects with metadata.
    """
    loader = DirectoryLoader(
        posts_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()

    # Enrich metadata from Jekyll front matter
    for doc in docs:
        filename = Path(doc.metadata["source"]).name
        # Parse date and slug from Jekyll filename format: YYYY-MM-DD-slug.md
        parts = filename.replace(".md", "").split("-", 3)
        if len(parts) == 4:
            doc.metadata["date"] = "-".join(parts[:3])
            doc.metadata["slug"] = parts[3]

    print(f"Loaded {len(docs)} documents")
    return docs
```

For richer sources — PDFs, Notion pages, GitHub repos — LangChain has loaders for all of them. The rest of the pipeline is identical regardless of source.

---

## Step 2: Text Splitting

Embedding models have token limits (typically 512 tokens for sentence transformers). Documents need to be split into chunks that fit within those limits, with some overlap between chunks to preserve context across boundaries:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks suitable for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,         # characters per chunk
        chunk_overlap=75,       # overlap between consecutive chunks
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],  # prefer splitting on headers
        length_function=len
    )

    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks")
    return chunks
```

The `separators` list is ordered by preference — the splitter tries to break on `##` headers first, then `###`, then double newlines, and so on. This produces chunks that tend to align with semantic sections rather than cutting mid-sentence.

The overlap parameter deserves attention. Without overlap, a concept that spans a chunk boundary would be split across two chunks that each lack enough context to be retrieved together. A 75-character overlap is conservative — for denser technical content, increasing this to 100–150 characters improves retrieval quality at the cost of slightly more storage.

---

## Step 3: Embedding and Vector Store

Embeddings convert text chunks into high-dimensional vectors where semantic similarity corresponds to geometric proximity. Similar chunks cluster together in the vector space, which is what enables similarity search:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_store(chunks: list[Document], persist_dir: str = "./chroma_db") -> Chroma:
    """
    Embed all chunks and store in a persistent Chroma vector database.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}   # normalize for cosine similarity
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    vectorstore.persist()
    print(f"Vector store built and persisted to {persist_dir}")
    return vectorstore


def load_vector_store(persist_dir: str = "./chroma_db") -> Chroma:
    """Load a previously built vector store from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
```

Chroma is a lightweight vector database that runs locally and persists to disk — ideal for a personal project or prototype. For production applications with larger corpora or multi-user concurrency requirements, Pinecone, Weaviate, or pgvector are more appropriate choices.

---

## Step 4: Building the RAG Chain

With documents indexed, the retrieval and generation pipeline is surprisingly concise with LangChain:

```python
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the provided context.

Use only the information in the context below to answer the question. If the context does not contain 
enough information to answer the question, say so clearly — do not make up information.

Context:
{context}

Question: {question}

Answer:"""


def build_rag_chain(vectorstore: Chroma) -> RetrievalQA:
    """
    Build a RetrievalQA chain that retrieves relevant chunks and generates answers.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0
    )

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}   # retrieve top 4 most similar chunks
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",           # "stuff" = concatenate all retrieved chunks into context
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True  # include retrieved chunks in response for transparency
    )

    return chain
```

The `temperature=0` setting is deliberate — for a retrieval-grounded application you want the model to be conservative and stick closely to the retrieved context, not generate creatively beyond it.

The `return_source_documents=True` flag is worth highlighting. Returning the source chunks alongside the answer lets you build a UI that shows users which documents the answer came from — a significant trust and debuggability improvement over a bare answer string.

---

## Step 5: Putting It Together

```python
def main():
    import os

    POSTS_DIR = "./_posts"
    VECTOR_STORE_DIR = "./chroma_db"

    # Build index (run once, or when new posts are added)
    if not os.path.exists(VECTOR_STORE_DIR):
        print("Building index...")
        docs = load_blog_posts(POSTS_DIR)
        chunks = split_documents(docs)
        vectorstore = build_vector_store(chunks, VECTOR_STORE_DIR)
    else:
        print("Loading existing index...")
        vectorstore = load_vector_store(VECTOR_STORE_DIR)

    # Build RAG chain
    chain = build_rag_chain(vectorstore)

    # Query loop
    print("\nKnowledge base ready. Ask a question (or 'quit' to exit):\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = chain({"query": question})
        print(f"\nAnswer: {result['result']}")
        print("\nSources:")
        for doc in result["source_documents"]:
            print(f"  - {doc.metadata.get('source', 'unknown')} (date: {doc.metadata.get('date', 'N/A')})")
        print()


if __name__ == "__main__":
    main()
```

Example interaction against a knowledge base built from this blog:

```
Question: How does the CareerPulse pipeline handle missing job categories?

Answer: Missing job categories are handled through a KNN-based imputation step in the 
Silver layer. TF-IDF vectorization is applied to the combined job title and description 
text, and a K-Nearest Neighbors classifier trained on labeled records assigns a predicted 
category to each NULL-category row. The trained classifier and vectorizer are serialized 
with joblib and registered in the Databricks workspace via MLflow for reproducibility.

Sources:
  - _posts/2026-03-17-CareerPulse_Category_Imputation_KNN.md (date: 2026-03-17)
```

---

## Limitations and Where to Go Next

**Retrieval quality is the bottleneck.** Generation with a capable LLM is the easier problem — if the retrieval step returns irrelevant chunks, no amount of prompt engineering will produce a good answer. Improving retrieval quality is where the highest returns are:

- **Hybrid search**: Combining dense (embedding) retrieval with sparse (BM25 keyword) retrieval catches cases where exact terminology matters more than semantic similarity.
- **Reranking**: A second-pass reranker model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) reorders the top-K retrieved chunks by relevance before passing them to the LLM.
- **Metadata filtering**: Adding date or category filters to the retriever reduces the search space and improves precision for structured queries.

**Production considerations.** For a multi-user deployment you'd want to move from Chroma to a hosted vector database, add authentication to the API layer, implement rate limiting, and build an incremental indexing pipeline so new documents are embedded and added to the store automatically rather than requiring a full rebuild.

**Evaluation.** The hardest part of RAG systems to get right is measuring whether they're actually working. A test set of question-answer pairs grounded in your document corpus, evaluated with metrics like RAGAS faithfulness and answer relevance, is the foundation of any serious RAG development workflow.

---

*LangChain · RAG · LLM · Vector Search · Applied ML · 2026*
