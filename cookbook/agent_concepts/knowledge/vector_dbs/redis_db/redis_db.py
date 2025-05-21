"""
1. Make sure to install sentence-transformers (uv pip install sentence-transformers) and pandas
"""

import pandas as pd
from agno.document import Document
from agno.vectordb.redis import Redis


"""
In this example we will process a csv files consisting of Quora like questions. The csv files consists of question1, question2 and similarity as columns.
We will embed only question1 column and use question2 column as query to check similarity.
Since each row in the csv files are independent information, we would want to avoid concatenating or chunking of information.
Currently, CSVReader doesn't offer an option to embed each row as a `Document` class, so we will process it here by ourselves.

Available operations:
1. Index Creation
2. Check if an index exists
3. Inserting documents as vectors into index
4. Dropping all keys in an index
5. Deleting an index
6. Search for a vector using a query embedding
"""


def create_document(csv_path):
    """
    Loads Quora dataset and drops nulls.
    """
    df = pd.read_csv(csv_path).dropna(subset=["question1", "question2"])
    df = df.head(100)  # Remove later - Only for testing

    documents = [Document(row.question1) for row in df.itertuples(index=False)]

    return documents


documents = create_document(
    "/home/yashm94/Projects/Redis/RedisVL/dataset/questions.csv"
)


vector_db = Redis(
    index_name="Quora",
    algorithm="hnsw",
    distance_metric="cosine",
    dims=384,
)

query = "I'm a 19-year-old. How can I improve my skills or what should I do to become an entrepreneur in the next few years?"

vector_db.drop()  # Dropping all keys from an existing index
if not vector_db.exists():  # Checking if an Index exists
    vector_db.create()  # Creating an Index
vector_db.insert(documents)  # Inserting documents/records into the index
results = vector_db.search(
    query=query, limit=3
)  # Searching for similar vector via query vector
for (
    result
) in results:  # printing top k values where k is the limit set in `search` above
    print(result.content)

"""
Test Redis integration with Agno's agentic framework.
"""

# from agno.knowledge.agent import AgentKnowledge
# from agno.agent import Agent
# from agno.vectordb.search import SearchType

# knowledge_base = AgentKnowledge(
#     vector_db=vector_db,
# )

# agent = Agent(
#     knowledge=knowledge_base,
#     search_knowledge=True,
#     show_tool_calls=True,
# )
# agent.print_response("How to make Thai curry?", markdown=True)
