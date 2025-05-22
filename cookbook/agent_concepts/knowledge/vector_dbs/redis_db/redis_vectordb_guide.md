# Redis VectorDB with Sentence Transformers: Indexing & Querying Quora-style Questions

This guide walks you through the process of setting up a Redis vector database to index and search Quora-style questions using the `sentence-transformers` library. You'll also learn how to integrate this with an agent like Agno for querying.

---

## 📦 Prerequisites

Ensure the following packages are installed using `uv`:

```bash
uv pip install sentence-transformers
uv pip install pandas
```

---

## 🐳 Setting Up Redis Server

### Run Redis Stack Server via Docker:

```bash
sudo docker run -d --name redis-vl -p <port>:<port> redis/redis-stack-server:latest
```

Replace `<port>` with the port you want to expose Redis on (default is `6379`).

### If the container is already created:

```bash
sudo docker start <container-id>
```

> **Note**: If a different port is used, update the `port` argument in the `Redis()` class accordingly.

---

## 📄 Dataset Format

We use a CSV file with the following columns:

- `question1`: The question to be embedded.
- `question2`: The question used as the query.
- `similarity`: (Optional) Similarity label between the questions.

Each row in the CSV is independent. Hence, avoid concatenating or chunking data for embedding.

Currently, `CSVReader` does not support embedding each row as a `Document`. This is handled manually.

---

## 🛠️ Supported Operations

- ✅ Create Index  
- ✅ Check if Index Exists  
- ✅ Insert Documents  
- ✅ Drop All Keys in an Index  
- ✅ Delete Index  
- ✅ Vector Search with Query Embedding  

---

## 🚀 Quickstart

### 1. Initialize Redis Vector DB

```python
vector_db = Redis(
    port=6379,
    host="localhost",
    index_name="Quora",
    algorithm="hnsw",
    distance_metric="cosine",
    dims=384,
)
```

### 2. Create a Knowledge Base from CSV

```python
knowledge_base = SingleCSVKnowledgeBase(
    path=<your/path>,
    vector_db=vector_db
)
```

### 3. Load Data into Vector DB

```python
knowledge_base.load(recreate=True, upsert=True, skip_existing=True)
```

- `recreate=True`: Deletes the existing index and recreates it.
- `upsert=True`: Inserts or updates vectors.
- `skip_existing=True`: Skips re-indexing already present documents.

---

## 🤖 Agno Agent Integration (Optional)

You can integrate the vector DB with an Agno Agent for semantic search:

```python
agent = Agent(
    model=OpenAIResponses(id="gpt-4o-mini"),
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
)

agent.print_response(
    "Retrieve closest sentence to `Why are rockets and boosters painted white?` from your knowledge base",
    markdown=True
)

agent.print_response(
    "Retrieve closest sentence to `How can I see all my Youtube comments?` from your knowledge base",
    markdown=True
)
```
