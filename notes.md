## PDF Project with Rag

- What is the Problem that RAG Solves
  - Basically RAG is what intermediate consults to data / documents information
  - The problem is that LLM does not have the context of my data
  - Also, I can not give a huge amount of information as context everytime

- RAG = Retrieval Aumented Generation with external knowledge sources

- If I simply send my documents as contexts, there are still some issues:
  - Cost (too many tokens everytime)
  - Limited Context Window

- There are 2 phases:
  1. Indexing Phase (providing data)
  2. Retrieval Data (chatting with data)

- Indexing Phase
  - Chucking data
  - Create Vector Embeddings
  - Vector DB

- Retrieval Phase
  - Convert the query into vector embeddings, so it retrieves only the specific data

- There are lots of vector DBs
  - Qdrant db is the recommended by the instructor

### Steps:

- Create a `docker-compose.yml` file:

  ```docker
    services:
      vector-db:
        image: qdrant/qdrant
        ports: - "6333:6333"
  ```

- Setup LangChain
  - LangChain is a open-source library that gives us a lot of utility
  - we are going to setup langchain and pypdf:
    - `pip install -U langchain-community pypdf`
