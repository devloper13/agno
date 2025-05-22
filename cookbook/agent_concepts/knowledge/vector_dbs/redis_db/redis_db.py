"""
Make sure to install following packages:
1. sentence-transformers: `uv pip install sentence-transformers`
2. Pandas: `uv pip install pandas`

Also, make sure to run the redis db server via the following command:
`sudo docker run -d --name redis-vl -p <port>:<port> redis/redis-stack-server:latest`

If you already have a container created use:
`sudo docker start <container-id>`

If you choose a differnet port, make sure to mention in the Redis() class defined below. By default it connects to 6379.
"""

import pandas as pd
from agno.document import Document
from agno.vectordb.redis import Redis
from agno.knowledge.singlecsv import SingleCSVKnowledgeBase
from agno.agent import Agent
from agno.models.openai.responses import OpenAIResponses

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

"""
Creating Redis vector db
"""

vector_db = Redis(
    port=6379,
    host="localhost",
    index_name="Quora",
    algorithm="hnsw",
    distance_metric="cosine",
    dims=384,
)

"""
Testing database operations via custom csv knowledgebase (loads only top 100 rows)
"""
knowledge_base = SingleCSVKnowledgeBase(
    path="/home/yashm94/Projects/Redis/RedisVL/dataset/questions.csv",
    vector_db=vector_db
)

knowledge_base.load(recreate=True, upsert=True, skip_existing=True)


"""
Testing Agno Agent Integration
"""

agent = Agent(
    model=OpenAIResponses(id="gpt-4o-mini"),
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
)

agent.print_response("Retrieve closest sentence to `Why are rockets and boosters painted white?` from your knowledge base", markdown=True)
agent.print_response("Retrieve closest sentence to `How can I see all my Youtube comments?` from your knowledge base", markdown=True)

