# Campus Knowledge Assistant

A learning project that answers questions using public university
documents and shows the sources supporting its answers.

## First version

- Work with 5–10 public Rutgers Computer Science documents.
- Retrieve passages relevant to a question.
- Generate answers using a local language model.
- Display source references and supporting passages.
- Indicate when the documents do not provide enough evidence.

## Architecture

### Preparing documents
1. Extract text while preserving source and page information.
2. Split text into smaller passages called chunks.
3. Convert chunks into numerical representations called embeddings.
4. Save the chunks, metadata, and embeddings.

### Answering questions
1. Convert the question into an embedding.
2. Retrieve passages with similar embeddings.
3. Give the question and retrieved passages to a local language model.
4. Display the answer alongside its sources.

## Planned tools

- Python: application logic and document processing.
- Sentence Transformers: generate embeddings.
- NumPy: compare embeddings for a small document collection.
- Ollama: run a language model locally.
- Streamlit: provide a simple user interface.

## Progress

### Day 1
- Created the project workspace and Python virtual environment.
- Added a .gitignore file.
- Learned the difference between retrieval and generation.
- Defined the initial scope and architecture.

## Status

Planning and setup. Document ingestion and answering are not implemented yet.
