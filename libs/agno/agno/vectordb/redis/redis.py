import numpy as np
from typing import Any, Dict, List, Optional
from agno.vectordb.base import VectorDb
from agno.embedder import Embedder
from agno.document import Document
from agno.utils.log import log_debug, log_info, logger
try:
    from redisvl.query import VectorQuery
    from redisvl.index import SearchIndex
except ImportError:
    raise ImportError(
        "The `redisvl` package is not installed. Please install it via `uv pip install redisvl`."
    )

class Redis(VectorDb):
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        host: Optional[str] = None,
        port: Optional[int] = 6379,
        index_name: Optional[str] = None,
        algorithm: Optional[str] = "hnsw",
        dims: Optional[int] = None,
        distance_metric: Optional[str] = "cosine"
    ):
        super().__init__()
        
        
        if embedder is None:
            from agno.embedder.sentence_transformer import SentenceTransformerEmbedder

            if dims is not None:
                self.dimensions = dims
                embedder = SentenceTransformerEmbedder(dims)
            else:
                embedder = SentenceTransformerEmbedder()
                self.dimensions = self.embedder.dimensions
            
            embedder.load_model()

            log_info("Embedder not provided, using SentenceTransformer as default.")
        self.embedder: Embedder = embedder
        

        if host is None:
            log_info("Since no host was provided, host will be set to localhost")
            self.host = "localhost"
        self.port = port
        self.redis_url = f"redis://{self.host}:{self.port}"
        log_info(f"The redis url is set as {self.redis_url}")
        
        self.index_name = index_name
        if self.index_name  is None:
            raise ValueError("Index name is mandatory.")
        
        self.algorithm = algorithm
        self.dims = dims
        self.distance_metric = distance_metric

        self.schema = self._create_schema()
        self.index = SearchIndex.from_dict(self.schema, redis_url=self.redis_url, validate_on_load=True)

    
    def _create_schema(self):
        """
        Fixed Schema assuming only two fields, data and it's embedding. Make sure to use the same fieldnames consistently everywhere.
        Changing this can cause broken changes.
        """
        schema = {
            'version': '0.1.0',
            'index': {
                'name': self.index_name,
                'prefix': self.index_name,
                'key_separator': ':',
                'storage_type': 'hash'
            },
            'fields': [
                {
                    'name': 'data',
                    'type': 'text'
                },
                {
                    'name': 'data_embedding',
                    'type': 'vector',
                    'attrs': {
                        'algorithm': self.algorithm,
                        'dims': self.dims,
                        'distance_metric': self.distance_metric,
                        'datatype': "float32"
                    }
                }
            ]
        }

        return schema
        
    def _format_search_response(self, results) -> List[Document]:
        """
        Changing the response format to List[Document] to make it compatible with Agno's requirements
        Warning: the fieldname `data` may change depeding on the return fieldnames mentioned in `search()` which as a result should be consistent with the schema defined in _create_Schema()
        """
        search_results: List[Document] = []
        for result in results:
            search_results.append(Document(content=result["data"]))
        
        return search_results

    def create(self) -> None:
        """
        Creates an index if it does not exist from the provided schema.
        """

        if not self.index.exists():
            self.index.create(overwrite=True, drop=False)
            log_info(f"Index `{self.index.name}` created.")
        else:
            log_info(f"Index `{self.index.name}` already exists.")

    
    async def async_create(self) -> None:
        return await super().async_create()
    

    def insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Populate data and it's embedding into the redis db
        The field names data and data_embedding was defined in the _create_schema() function. Any changes there should reflect here.
        """
        records = []
        for document in documents:
            document.embed(embedder=self.embedder)
            records.append(
                {
                    "data": document.content,
                    "data_embedding":  np.array(document.embedding,dtype=np.float32).tobytes(),
                }
            )
    
        self.index.load(records)
        num_docs = self.index.info()['num_docs']
        log_info(f"{num_docs} docs inserted")
    
    async def async_insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        return await super().async_insert(documents, filters)
    
    def upsert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Updating exising index with additional data
        """
        self.insert(documents)
    
    async def async_upsert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        return await super().async_upsert(documents, filters)
    

    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Performs vector search in Redis given an embedding.
        THe fieldnames data_embedding and data were defined in schema in _create_schema(). Make sure any changes there should reflect here
        """
        query_embedding = self.embedder.get_embedding(query)
        query = VectorQuery(
            vector=query_embedding,
            vector_field_name="data_embedding",
            return_fields=["data"],
            num_results=limit
        )
        results = self.index.query(query)
        return self._format_search_response(results)

    
    async def async_search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        return await super().async_search(query, limit, filters)
    
    def delete(self) -> bool:
        try:   
            self.index.delete(drop=True)
            log_info("Index deleted")
        except:
            log_info("Something went wrong in deleting the index")
            return False
        return True
    
    def drop(self) -> None:
        return self.index.create(overwrite=True, drop=True)
    
    async def async_drop(self) -> None:
        return await super().async_drop()
    

    def exists(self) -> bool:

        if not self.index.exists():
            return False
        return True
    
    async def async_exists(self) -> bool:
        return await super().async_exists()
        
    def doc_exists(self, document: Document) -> bool:
        return super().doc_exists(document)

    async def async_doc_exists(self, document: Document) -> bool:
        return await super().async_doc_exists(document)

   
    
    def name_exists(self, name: str) -> bool:
        return super().name_exists(name)
    
    async def async_name_exists(self, name: str) -> bool:
        return super().async_name_exists(name)
    
    

    
    