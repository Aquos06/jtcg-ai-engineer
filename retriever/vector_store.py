import asyncio
import logging
from typing import Any, Optional, List

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index.core.schema import ObjectType
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

from retriever.milvus import CustomMilvusVector
from config.env import EMBED_DIM, MILVUS_URL, COLLECTION_NAME

logger = logging.getLogger(__name__)


class CustomVectorIndexRetriever(VectorIndexRetriever):
    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        filters: Optional[MetadataFilters] = None,
        alpha: Optional[float] = None,
        node_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        sparse_top_k: Optional[int] = None,
        hybrid_top_k: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        embed_model: Optional[BaseEmbedding] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index=index,
            similarity_top_k=similarity_top_k,
            vector_store_query_mode=vector_store_query_mode,
            filters=filters,
            alpha=alpha,
            node_ids=node_ids,
            doc_ids=doc_ids,
            sparse_top_k=sparse_top_k,
            hybrid_top_k=hybrid_top_k,
            callback_manager=callback_manager,
            object_map=object_map,
            embed_model=embed_model,
            verbose=verbose,
            **kwargs,
        )

    def _determine_nodes_to_fetch(
        self, query_result: VectorStoreQueryResult
    ) -> List[str]:
        if query_result.nodes:
            return [
                node.node_id
                for node in query_result.nodes
                if node.as_related_node_info().node_type
                not in [ObjectType.TEXT, ObjectType.INDEX]
            ]
        elif query_result.ids:
            return [
                self._index.index_struct.nodes_dict[idx] for idx in query_result.ids
            ]
        else:
            raise ValueError(
                "Vector store query result should return at least one of nodes or ids."
            )


class CustomVectorStoreIndex(VectorStoreIndex):
    def __init__(
        self,
        vector_store: BasePydanticVectorStore,
        embed_model: Optional[EmbedType] = None,
        index_struct: Optional[IndexDict] = None,
        **kwargs: Any,
    ):
        kwargs.pop("nodes", None)
        kwargs.pop("storage_context", None)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        super().__init__(
            nodes=None if index_struct else [],
            embed_model=embed_model,
            storage_context=storage_context,
            index_struct=index_struct,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        return CustomVectorIndexRetriever(
            self,
            node_ids=list(self.index_struct.nodes_dict.values()),
            callback_manager=self._callback_manager,
            object_map=self._object_map,
            **kwargs,
        )
    
def get_vector_store(collection_name: str) -> Optional[CustomMilvusVector]:
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        vector_store = CustomMilvusVector(
            uri=MILVUS_URL,
            collection_name=collection_name,
            dim=EMBED_DIM
        )

        if vector_store.dimension != EMBED_DIM:
            raise ValueError(
                f"Dimension mismatch: vector_store has dimension {vector_store.dimension}, "
                f"but embeded_model has dimension {EMBED_DIM}. Ensure both are aligned."
            )

        return vector_store
    except Exception as e:
        logger.error(
            f"Milvus vector store '{collection_name}' not found. Exception: {e}",
            exc_info=True,
        )
        raise e


def create_vector_store(collection_name: str) -> Optional[CustomMilvusVector]:
    try:
        vector_store = CustomMilvusVector(
            uri=MILVUS_URL,
            collection_name=collection_name,
            dim=EMBED_DIM,
        )

        return vector_store
    except Exception as e:
        logger.error(f"Failed to create Milvus vector store: {e}", exc_info=True)
        raise e

milvus_vector_store = get_vector_store(collection_name=COLLECTION_NAME)