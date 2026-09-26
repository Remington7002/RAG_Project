# Retrieval-Augmented Generation (RAG) Project

A Retrieval-Augmented Generation (RAG) pipeline built as part of an AI/ML internship. The project focuses on preparing a large-scale crypto-news dataset for downstream semantic search, vector indexing, and Retrieval-Augmented Generation.

## Project Overview

This project processes a large collection of cryptocurrency news articles and prepares the data for a RAG system.

The current pipeline includes:

* Environment verification
* Large-scale dataset preparation
* HTML cleaning
* Unicode normalization
* Whitespace cleanup
* Language filtering
* JSONL-based corpus generation
* Automated preprocessing tests
* Python virtual environment
* CPU-based machine learning environment

The project is designed to continue toward document chunking, embedding generation, vector database indexing, retrieval, and LLM-based response generation.

---

## Dataset

The project uses the **Crypto News / CoinDesk 2020–2025** dataset from Hugging Face.

Dataset source:

`maryamfakhari/crypto-news-coindesk-2020-2025`

The downloaded dataset contains approximately:

* **229,000+ news documents**
* Original format: CSV
* Fields include:

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

The raw dataset is stored locally and is excluded from Git using `.gitignore`.

---

## Project Pipeline

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
       +--> HTML Removal
       +--> Unicode Normalization
       +--> Whitespace Cleanup
       +--> Language Filtering
       |
       v
Clean Corpus
       |
       v
Document Chunking
       |
       v
Embeddings
       |
       v
ChromaDB Vector Database
       |
       v
Retriever
       |
       v
LLM / RAG Response
```

---

## Project Structure

```text
RAG_Project/
│
├── data/
│   ├── raw/
│   │   └── Document_5000.csv
│   │
│   └── processed/
│       └── clean_corpus_1.jsonl
│
├── scripts/
│   └── verify_env.py
│
├── src/
│   ├── __init__.py
│   └── preprocessing.py
│
├── tests/
│   └── test_preprocessing.py
│
├── prepare_dataset.py
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technologies Used

* **Python 3.12**
* **Pandas**
* **NumPy**
* **Sentence Transformers**
* **ChromaDB**
* **spaCy**
* **PyTest**
* **JSONL**
* **Git & GitHub**

---

## Environment

The project uses a Python virtual environment to isolate dependencies.

Python version:

```text
Python 3.12.7
```

The current environment was configured for CPU-based execution.

GPU verification confirms that CUDA is not available on the current machine.

---

## Environment Verification

The project includes:

```text
scripts/verify_env.py
```

This script verifies the required environment and libraries.

The environment verification successfully confirms the availability of:

* NumPy
* Pandas
* Sentence Transformers
* ChromaDB
* spaCy
* Python environment

---

## Data Preparation

The raw CSV dataset is converted into a JSONL corpus using:

```text
prepare_dataset.py
```

The generated corpus contains the news documents in a format suitable for subsequent NLP processing.

---

## Text Preprocessing

The preprocessing pipeline is implemented in:

```text
src/preprocessing.py
```

It performs the following operations:

### 1. HTML Stripping

Removes HTML tags and unwanted markup from article content.

### 2. Unicode Normalization

Normalizes Unicode characters to provide consistent text representation.

### 3. Whitespace Cleanup

Removes unnecessary spaces, line breaks, and formatting artifacts.

### 4. Language Filtering

Filters documents according to the preprocessing requirements.

### 5. Clean Corpus Generation

The processed documents are stored as JSONL for the next stage of the RAG pipeline.

---

## Testing

Automated tests are implemented using PyTest.

Run:

```bash
pytest -q
```

Current test status:

```text
13 passed
```

The tests cover the preprocessing functionality and help ensure that changes to the pipeline do not break existing behavior.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Remington7002/RAG_Project.git
```

Enter the project directory:

```bash
cd RAG_Project
```

Create a virtual environment:

```bash
py -3.12 -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Verify Environment

```bash
python scripts/verify_env.py
```

### Prepare Dataset

```bash
python prepare_dataset.py
```

### Run Preprocessing

```bash
python src/preprocessing.py
```

### Run Tests

```bash
pytest -q
```

---

## Current Progress

### Completed

* [x] Python virtual environment
* [x] Python 3.12 environment
* [x] Dependency installation
* [x] Environment verification
* [x] Large crypto-news dataset downloaded
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

### Next Steps

* [ ] Document chunking
* [ ] Chunk quality testing
* [ ] Sentence-transformer embeddings
* [ ] ChromaDB vector indexing
* [ ] Semantic similarity search
* [ ] Retriever implementation
* [ ] RAG prompt construction
* [ ] LLM integration
* [ ] End-to-end RAG pipeline
* [ ] Retrieval evaluation
* [ ] Final documentation

---

## Git Workflow

After making changes:

```bash
git status
```

Add the changes:

```bash
git add .
```

Commit:

```bash
git commit -m "Update RAG preprocessing pipeline"
```

Push to GitHub:

```bash
git push origin main
```

If the remote repository contains changes that are not available locally, synchronize first:

```bash
git pull origin main --allow-unrelated-histories
```

Then push:

```bash
git push origin main
```

---

## Repository

GitHub:

**Remington7002/RAG_Project**

The repository contains the source code, preprocessing pipeline, tests, configuration, and project documentation. Large datasets and environment-specific files are excluded from version control through `.gitignore`.

---

## Purpose

The purpose of this project is to build a practical understanding of the complete RAG workflow:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Document Processing
      ↓
Chunking
      ↓
Embedding
      ↓
Vector Storage
      ↓
Retrieval
      ↓
Generation
```

This project serves as the foundation for implementing a complete Retrieval-Augmented Generation system over a large cryptocurrency news corpus.
