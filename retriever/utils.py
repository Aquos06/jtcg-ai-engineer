from pydantic import UUID4
from typing import List, Optional

from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.schema import TextNode

from retriever.const import EMBEDDING_TOP_K
from retriever.vector_store import CustomVectorStoreIndex
from retriever.vector_store import milvus_vector_store
from retriever.embedding import embedding_model

def get_retrieval_engine() -> BaseRetriever:
    vector_index = CustomVectorStoreIndex(
        vector_store=milvus_vector_store,
        embed_model=embedding_model,
        insert_batch_size=512,
    )

    vector_retriever = vector_index.as_retriever(
        similarity_top_k=EMBEDDING_TOP_K,
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
    )

    return vector_retriever

def add_node(text:str, node_id: UUID4, metadata: Optional[dict] = {}) -> None:
    node_to_insert = TextNode(
        id_=str(node_id), text=text, metadata=metadata
    )
    milvus_vector_store.add([node_to_insert])

def add_node_batch(nodes: List[TextNode]) -> None:
    milvus_vector_store.add(nodes=nodes)

def retreive_from_vector_store(text: str) -> List[NodeWithScore]:
    retrieval_engine = get_retrieval_engine()
    return retrieval_engine.retrieve(text)

