## PDF Chat with RAG

RAG = Retrieval Augmented Generation.

It solves this problem: the LLM does not automatically know the content of my private files, and sending the full PDF in every prompt would be expensive, slow, and limited by the model context window.

So instead of sending the whole document every time, we:

1. Index the document once
2. Retrieve only the most relevant chunks for each user question
3. Send those chunks as context to the LLM
4. Ask the LLM to answer based only on that context

---

## Mental Model

```text
PDF
→ pages
→ text chunks
→ embeddings
→ vector database
→ user question
→ question embedding
→ similar chunks
→ LLM answer with context
```

---

## Main Components

- `PyPDFLoader`
  - Loads the PDF and converts each page into a LangChain document

- `RecursiveCharacterTextSplitter`
  - Splits large pages into smaller text chunks
  - `chunk_size=1000`: each chunk has around 1000 characters
  - `chunk_overlap=400`: part of the previous chunk is repeated in the next chunk to preserve context

- `OpenAIEmbeddings`
  - Converts text into vectors/numbers that represent semantic meaning
  - Similar meanings should produce similar vectors

- `QdrantVectorStore`
  - Stores the chunks and their embeddings
  - Allows similarity search: “find chunks semantically close to this question”

- `OpenAI`
  - Receives the retrieved context + user question
  - Generates the final answer

---

## Indexing Phase (`index.py`)

This phase prepares the PDF for search.

```text
Load PDF
→ split into chunks
→ create embeddings
→ store in Qdrant collection
```

Important line:

```python
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url=VECTOR_STORE_URL,
    collection_name="learning_rag",
)
```

Meaning:

- `documents=chunks`: text chunks that will be stored
- `embedding=embedding_model`: model used to convert each chunk into vectors
- `url=VECTOR_STORE_URL`: where Qdrant is running locally
- `collection_name="learning_rag"`: name of the vector DB collection/table-like storage

This script only needs to run when indexing or re-indexing the PDF.

---

## Retrieval Phase (`chat.py`)

This phase answers user questions using the indexed PDF.

```text
User asks a question
→ convert question into embedding
→ search similar chunks in Qdrant
→ build context from retrieved chunks
→ send context + question to LLM
→ print answer
```

Important line:

```python
search_results = vector_db.similarity_search(query=user_query)
```

Meaning:

- The user question is converted into an embedding
- Qdrant finds the most semantically similar chunks from the PDF
- Those chunks become the context for the LLM

The LLM should not answer from general knowledge. It should answer only from the retrieved PDF context.

---

## Docker / Qdrant

Qdrant is the vector database used in this project.

We run it locally with Docker so we do not need to manually install Qdrant on the machine.

`docker-compose.yml`:

```yaml
services:
  vector-db:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
```

Run:

```bash
docker compose up
```

Qdrant runs at:

```text
http://localhost:6333
```

---

## Dependencies

```bash
pip install -U langchain-community pypdf
pip install -U langchain-text-splitters
pip install -U langchain-openai
pip install -U langchain-qdrant
```

---

## Big Picture

This project is not “the AI reads the PDF directly.”

It is:

```text
The app searches the PDF first, retrieves only the relevant parts, and gives those parts to the LLM so it can answer with document context.
```

That is the core idea of RAG.
