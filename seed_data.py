import uuid
import pandas as pd

from tqdm import tqdm
from typing import List
from llama_index.core.schema import TextNode

from retriever.utils import add_node_batch

def load_data_and_build_retrievers():
    """Loads all data sources and prepares them for the tools."""
    node_list :List[TextNode] = []

    knowledge_df = pd.read_csv("document/knowledge_base.csv")
    for _, row in tqdm(knowledge_df.iterrows(), desc="Adding knowledge base to vector store"):
        doc_content = f"Title: {row['title']}\nContent: {row['content']}\nTags: {row.get('tags/0', '')}, {row.get('tags/1', '')}, {row.get('tags/2', '')}"
        metadata={
            "doc_id": row['id'],
            "title": row['title'],
            "content": row['content'],
            "url": row['urls/0/href'],
            "image": row['images/0'],
            "tag": [row['tags/0'], row['tags/1'], row['tags/2']]
        }
        node_list.append(
            TextNode(
                id_=str(uuid.uuid4()),
                text=doc_content,
                metadata=metadata
            )
        )
    add_node_batch(nodes=node_list)

if __name__ == "__main__":
    load_data_and_build_retrievers()