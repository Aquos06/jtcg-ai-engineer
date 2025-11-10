import logging
from typing import Any, List, Optional, Tuple

import jieba
from llama_index.core.schema import TextNode
from llama_index.core.utils import iter_batch
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.milvus import MilvusVectorStore as MilvusVectorStoreBase
from llama_index.vector_stores.milvus.base import MILVUS_ID_FIELD
from pymilvus import (
    AnnSearchRequest,
    Collection,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)

from retriever.embedding import embedding_model

logger = logging.getLogger(__name__)

def fakefunction():
    pass


class CustomMilvusVector(MilvusVectorStoreBase):
    text_field: str = "text"
    sparse_function_name: str = "text_bm25"
    doc_id_field: str = "doc_id"

    def __init__(
        self,
        uri: str,
        collection_name: str,
        dim: Optional[int] = None,
    ) -> None:
        super().__init__(
            collection_name=collection_name,
            dim=dim,
            uri=uri,
            enable_sparse=True,
        )

    @property
    def dimension(self):
        for field in self._collection.schema.fields:
            if field.name == "embedding":
                return field.params["dim"]
        return None

    def _create_hybrid_index(self, collection_name: str) -> None:
        if collection_name not in self.client.list_collections():
            schema = MilvusClient.create_schema(
                auto_id=False, enable_dynamic_field=True
            )
            schema.add_field(
                field_name=MILVUS_ID_FIELD,
                datatype=DataType.VARCHAR,
                max_length=65535,
                is_primary=True,
            )
            schema.add_field(
                field_name=self.doc_id_field,
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            schema.add_field(
                field_name=self.embedding_field,
                datatype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            )
            schema.add_field(
                field_name=self.sparse_embedding_field,
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            schema.add_field(
                field_name=self.text_field,
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
            )

            bm25_function = Function(
                name=self.sparse_function_name,
                input_field_names=[self.text_field],
                output_field_names=[self.sparse_embedding_field],
                function_type=FunctionType.BM25,
            )

            schema.add_function(bm25_function)
            self.client.create_collection(
                collection_name=collection_name, schema=schema
            )

        self._collection = Collection(collection_name, using=self.client._using)

        dense_index_exists = self._collection.has_index(index_name=self.embedding_field)
        sparse_index_exists = self._collection.has_index(
            index_name=self.sparse_embedding_field
        )

        if dense_index_exists:
            self._collection.release()
            self._collection.drop_index(index_name=self.embedding_field)

        if sparse_index_exists:
            self._collection.release()
            self._collection.drop_index(index_name=self.sparse_embedding_field)

        # Create dense index
        base_params = self.index_config.copy()
        index_type = base_params.pop("index_type", "FLAT")
        dense_index = {
            "params": base_params,
            "metric_type": self.similarity_metric,
            "index_type": index_type,
        }
        self._collection.create_index(self.embedding_field, dense_index)

        # Create sparse index
        sparse_index = {
            "field_name": self.sparse_embedding_field,
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
            "params": {
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
        }
        self._collection.create_index(self.sparse_embedding_field, sparse_index)
        self._collection.load()

    def do_jieba(self, text: str) -> str:
        tokenized_query = jieba.cut(text)
        filtered_query = [token for token in tokenized_query if token.strip() != ""]
        text = " ".join(filtered_query)
        return text

    def add(self, nodes: List[TextNode], **add_kwargs: Any) -> List[str]:
        insert_list = []
        insert_ids = []

        for node in nodes:
            entry = node_to_metadata_dict(node)
            entry[MILVUS_ID_FIELD] = node.node_id
            entry[self.embedding_field] = embedding_model.get_text_embedding(node.text)
            entry[self.text_field] = self.do_jieba(node.text)
            entry[self.doc_id_field] = str(node.metadata.get("doc_id", ""))

            insert_ids.append(node.node_id)
            insert_list.append(entry)

        for insert_batch in iter_batch(insert_list, self.batch_size):
            self.client.insert(self.collection_name, insert_batch)

        if add_kwargs.get("force_flush", False):
            self.client.flush(self.collection_name)

        return insert_ids

    async def async_add(self, nodes: List[TextNode], **add_kwargs: Any) -> List[str]:
        insert_list = []
        insert_ids = []

        for node in nodes:
            entry = node_to_metadata_dict(node)
            entry[MILVUS_ID_FIELD] = node.node_id
            entry[self.embedding_field] = node.embedding
            entry[self.text_field] = self.do_jieba(node.text)
            entry[self.doc_id_field] = node.metadata.get("doc_id", "")

            insert_ids.append(node.node_id)
            insert_list.append(entry)

        for insert_batch in iter_batch(insert_list, self.batch_size):
            await self.aclient.insert(self.collection_name, insert_batch)

        if add_kwargs.get("force_flush", False):
            raise NotImplementedError("force flush is not supported in async mode")

        return insert_ids

    def _hybrid_search(
        self, query: VectorStoreQuery, string_expr: str, output_fields: List[str]
    ) -> Tuple[List[TextNode], List[float], List[str]]:
        sparse_req = AnnSearchRequest(
            data=[self.do_jieba(query.query_str)],
            anns_field=self.sparse_embedding_field,
            param={"metric_type": "BM25"},
            limit=query.similarity_top_k,
            expr=string_expr,
        )

        dense_req = AnnSearchRequest(
            data=[query.query_embedding],
            anns_field=self.embedding_field,
            param={"metric_type": self.similarity_metric, "params": self.search_config},
            limit=query.similarity_top_k,
            expr=string_expr,
        )

        if self.hybrid_ranker == "WeightedRanker":
            if self.hybrid_ranker_params == {}:
                self.hybrid_ranker_params = {"weights": [1.0, 1.0]}
            ranker = WeightedRanker(*self.hybrid_ranker_params["weights"])
        elif self.hybrid_ranker == "RRFRanker":
            if self.hybrid_ranker_params == {}:
                self.hybrid_ranker_params = {"k": 60}
            ranker = RRFRanker(self.hybrid_ranker_params["k"])
        else:
            raise ValueError(f"Unsupported ranker: {self.hybrid_ranker}")

        res = self.client.hybrid_search(
            self.collection_name,
            [dense_req, sparse_req],
            ranker=ranker,
            limit=query.similarity_top_k,
            output_fields=output_fields,
        )

        nodes, similarities, ids = self._parse_from_milvus_results(res)

        return nodes, similarities, ids

    async def _async_hybrid_search(
        self,
        query: VectorStoreQuery,
        string_expr: str,
        output_fields: List[str],
        **kwargs,
    ) -> Tuple[List[TextNode], List[float], List[str]]:
        sparse_req = AnnSearchRequest(
            data=[query.query_str],
            anns_field=self.sparse_embedding_field,
            param={"metric_type": "BM25"},
            limit=query.similarity_top_k,
            expr=string_expr,
        )

        dense_req = AnnSearchRequest(
            data=[query.query_embedding],
            anns_field=self.embedding_field,
            param={"metric_type": self.similarity_metric, "params": self.search_config},
            limit=query.similarity_top_k,
            expr=string_expr,
        )

        if self.hybrid_ranker == "WeightedRanker":
            if self.hybrid_ranker_params == {}:
                self.hybrid_ranker_params = {"weights": [1.0, 1.0]}
            ranker = WeightedRanker(*self.hybrid_ranker_params["weights"])
        elif self.hybrid_ranker == "RRFRanker":
            if self.hybrid_ranker_params == {}:
                self.hybrid_ranker_params = {"k": 60}
            ranker = RRFRanker(self.hybrid_ranker_params["k"])
        else:
            raise ValueError(f"Unsupported ranker: {self.hybrid_ranker}")

        res = await self.aclient.hybrid_search(
            self.collection_name,
            [dense_req, sparse_req],
            ranker=ranker,
            limit=query.similarity_top_k,
            output_fields=output_fields,
        )

        nodes, similarities, ids = self._parse_from_milvus_results(res)
        return nodes, similarities, ids

