# Retrieval-Augmented Generation (RAG) Project

A Retrieval-Augmented Generation (RAG) pipeline developed as part of an AI/ML internship. The project processes a large-scale cryptocurrency news dataset and implements the core stages of a RAG system, including data preprocessing, document chunking, embedding generation, vector database ingestion, semantic search, and retrieval benchmarking.

---

## Project Overview

This project builds a complete foundation for a Retrieval-Augmented Generation system using a large cryptocurrency news corpus.

The pipeline currently covers:

* Environment setup and verification
* Large-scale dataset preparation
* Text preprocessing
* HTML removal
* Unicode normalization
* Whitespace cleanup
* Language filtering
* JSONL corpus generation
* Document chunking
* Sentence-Transformer embeddings
* Embedding benchmarking
* ChromaDB vector database ingestion
* Semantic similarity search
* Retrieval benchmarking
* Automated unit testing

The next stage is to connect the retrieval pipeline to an LLM and build the complete end-to-end RAG question-answering system.

---

## Dataset

The project uses the **Crypto News / CoinDesk 2020–2025** dataset from Hugging Face.

Dataset:

`maryamfakhari/crypto-news-coindesk-2020-2025`

The dataset contains approximately **229,000+ cryptocurrency news documents**.

### Original Dataset Format

The raw dataset is provided as a CSV file containing fields such as:

* `id`
* `guid`
* `published_on`
* `title`
* `body`
* `url`
* `imageurl`
* `tags`
* `categories`
* `source`
* `upvotes`
* `downvotes`
* `last_update`

The raw dataset is intentionally excluded from GitHub through `.gitignore` because of its size.

---

## RAG Pipeline

The implemented pipeline follows this workflow:

```text
Raw CSV Dataset
       |
       v
Dataset Preparation
       |
       v
JSONL Corpus
       |
       v
Text Preprocessing
       |
       +---- HTML Removal
       +---- Unicode Normalization
       +---- Whitespace Cleanup
       +---- Language Filtering
       |
       v
Clean Corpus
       |
       v
Document Chunking
       |
       v
Sentence-Transformer Embeddings
       |
       v
ChromaDB Vector Database
       |
       v
Semantic Search
       |
       v
Document Retrieval
       |
       v
Retrieval Benchmarking
       |
       v
LLM
       |
       v
RAG Response
```

The data preparation, preprocessing, chunking, embedding, vector storage, semantic search, and retrieval benchmarking stages have been implemented.

---

# Project Structure

```text
RAG_Project/
│
├── data/
│   ├── raw/
│   │   └── Document_5000.csv
│   │
│   └── processed/
│       ├── crypto_news.jsonl
│       └── clean_corpus_1.jsonl
│
├── scripts/
│   ├── verify_env.py
│   ├── generate_embeddings.py
│   ├── ingest_chroma.py
│   ├── semantic_search.py
│   ├── benchmark_embeddings.py
│   └── benchmark_retrieval.py
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── chunking.py
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_chunking.py
│
├── prepare_dataset.py
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

> Large datasets, generated vector databases, virtual environments, and other local artifacts are excluded from version control where appropriate.

---

# Technologies Used

* **Python 3.12**
* **Pandas**
* **NumPy**
* **Sentence Transformers**
* **ChromaDB**
* **spaCy**
* **PyTest**
* **JSONL**
* **Git**
* **GitHub**

---

# Environment

The project runs inside an isolated Python virtual environment.

### Python Version

```text
Python 3.12.7
```

### Hardware

The current development environment uses CPU-based processing.

CUDA/GPU acceleration is not available on the current machine.

The environment verification script checks the installed dependencies and available hardware configuration.

---

# Environment Verification

The project includes:

```text
scripts/verify_env.py
```

Run it with:

```bash
python scripts/verify_env.py
```

The script verifies the project environment and required machine-learning libraries, including:

* NumPy
* Pandas
* Sentence Transformers
* ChromaDB
* spaCy
* PyTorch

---

# Data Preparation

The raw cryptocurrency news dataset is converted into a JSONL corpus using:

```text
prepare_dataset.py
```

This converts the original CSV dataset into a document-oriented format that can be processed efficiently by the NLP pipeline.

---

# Text Preprocessing

The preprocessing pipeline is implemented in:

```text
src/preprocessing.py
```

The preprocessing stage performs several operations.

### HTML Removal

Removes HTML tags and unwanted markup from the news articles.

### Unicode Normalization

Normalizes Unicode characters to provide consistent text representation.

### Whitespace Cleanup

Removes unnecessary spaces, line breaks, and formatting artifacts.

### Language Filtering

Filters documents according to the project's language-processing requirements.

### Clean Corpus

The resulting documents are stored in JSONL format for downstream processing.

---

# Document Chunking

Document chunking is implemented in:

```text
src/chunking.py
```

Long news articles are divided into smaller text chunks before embedding.

Chunking is important for RAG because smaller, meaningful sections allow the retrieval system to return more relevant portions of documents rather than retrieving entire long articles.

The chunking implementation is covered by automated tests in:

```text
tests/test_chunking.py
```

---

# Embedding Generation

Embeddings are generated using **Sentence Transformers**.

The implementation is located in:

```text
scripts/generate_embeddings.py
```

The embedding stage converts text chunks into numerical vector representations.

These vectors allow the system to compare documents based on semantic similarity rather than simple keyword matching.

---

# Embedding Benchmarking

Embedding performance is evaluated using:

```text
scripts/benchmark_embeddings.py
```

The benchmark is used to evaluate the embedding pipeline and provide performance information about the embedding generation process.

---

# ChromaDB Vector Database

The generated embeddings are stored in **ChromaDB**.

The ingestion pipeline is implemented in:

```text
scripts/ingest_chroma.py
```

The vector database provides persistent storage and enables efficient similarity-based retrieval of relevant document chunks.

---

# Semantic Search

Semantic search is implemented in:

```text
scripts/semantic_search.py
```

Instead of searching only for exact keywords, the system converts the search query into an embedding and retrieves documents that are semantically similar to the query.

Conceptually:

```text
User Query
     |
     v
Query Embedding
     |
     v
ChromaDB
     |
     v
Similarity Search
     |
     v
Relevant Chunks
```

---

# Retrieval Benchmarking

Retrieval performance is evaluated using:

```text
scripts/benchmark_retrieval.py
```

This allows the retrieval stage to be tested and benchmarked before connecting it to the generation component of the RAG system.

---

# Testing

Automated tests are implemented using **PyTest**.

Run:

```bash
pytest -q
```

The project currently contains automated tests covering preprocessing and document chunking.

Current test status:

```text
13 passed
```

Tests help ensure that changes to the preprocessing and chunking pipeline do not introduce regressions.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Remington7002/RAG_Project.git
```

## 2. Enter the Project Directory

```bash
cd RAG_Project
```

## 3. Create the Virtual Environment

```bash
py -3.12 -m venv venv
```

## 4. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Verify the Environment

```bash
python scripts/verify_env.py
```

## Prepare the Dataset

```bash
python prepare_dataset.py
```

## Run Preprocessing

```bash
python src/preprocessing.py
```

## Run Tests

```bash
pytest -q
```

## Generate Embeddings

```bash
python scripts/generate_embeddings.py
```

## Ingest Embeddings into ChromaDB

```bash
python scripts/ingest_chroma.py
```

## Perform Semantic Search

```bash
python scripts/semantic_search.py
```

## Benchmark Embeddings

```bash
python scripts/benchmark_embeddings.py
```

## Benchmark Retrieval

```bash
python scripts/benchmark_retrieval.py
```

---

# Project Progress

## Week 1 — Environment & Data Preparation

* [x] Python virtual environment
* [x] Python 3.12 environment
* [x] Dependency installation
* [x] Environment verification
* [x] CPU environment verification
* [x] Large cryptocurrency news dataset downloaded
* [x] 229K+ documents prepared
* [x] CSV → JSONL conversion
* [x] HTML stripping
* [x] Unicode normalization
* [x] Whitespace cleanup
* [x] Language filtering
* [x] Clean corpus generation
* [x] Automated preprocessing tests
* [x] Git repository setup
* [x] GitHub repository setup

## Week 2 — RAG Processing & Retrieval

* [x] Document chunking
* [x] Chunk quality testing
* [x] Sentence-Transformer embeddings
* [x] Embedding generation
* [x] Embedding benchmarking
* [x] ChromaDB vector database setup
* [x] Document ingestion into ChromaDB
* [x] Semantic similarity search
* [x] Retriever implementation
* [x] Retrieval benchmarking
* [x] Chunking automated tests


---

# Git Workflow

Check the current repository state:

```bash
git status
```

Stage the complete project:

```bash
git add .
```

Commit changes:

```bash
git commit -m "Update complete RAG pipeline"
```

Push to GitHub:

```bash
git push origin main
```

If the remote repository contains commits that are not available locally:

```bash
git pull origin main --allow-unrelated-histories
```

Then push again:

```bash
git push origin main
```

---

# GitHub Repository

**Remington7002/RAG_Project**

The repository contains the RAG source code, preprocessing pipeline, chunking implementation, embedding scripts, ChromaDB ingestion, semantic search, benchmarking scripts, automated tests, configuration, and project documentation.

Large datasets and local environment files are intentionally excluded from GitHub.

---

# Project Goals

The main goal of this project is to understand and implement the major components of a Retrieval-Augmented Generation system.

The overall architecture is:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Document Processing
      ↓
Document Chunking
      ↓
Embedding Generation
      ↓
Vector Storage
      ↓
Semantic Retrieval
      ↓
Context Construction
      ↓
LLM Generation
      ↓
RAG Response
```

The project currently has the **data processing, chunking, embedding, vector storage, semantic search, and retrieval stages completed**.

The remaining work focuses on connecting the retrieval system to an LLM and producing context-aware generated responses.

---

# Author

**Ahmed Raza**

AI/ML Internship — Retrieval-Augmented Generation Project
