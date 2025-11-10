import uuid
import pandas as pd

from tqdm import tqdm
from typing import List
from llama_index.core.schema import TextNode

from retriever.utils import add_node_batch, add_product_node_batch

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

def seed_products_db(csv_path: str = "document/product.csv"):
    """
    Loads products.csv, transforms each product into a searchable TextNode,
    and adds them to the vector store.
    """
    node_list: List[TextNode] = []

    products_df = pd.read_csv(csv_path).fillna('')

    for _, row in tqdm(products_df.iterrows(), desc="Adding products to vector store"):
        includes = ", ".join(filter(None, [
            row.get('specs/includes/0', ''),
            row.get('specs/includes/1', ''),
            row.get('specs/includes/2', '')
        ]))

        doc_content = (
            f"Product Name: {row.get('name', '')}\n"
            f"Type: {row.get('specs/arm_type', '')}\n"
            f"Compatibility Notes: {row.get('compatibility_notes', '')}\n"
            f"Key Features: USB Hub ({row.get('specs/usb_hub', 'false')}), "
            f"Rotation ({row.get('specs/rotation', 'N/A')}), "
            f"Tilt ({row.get('specs/tilt', 'N/A')})\n"
            f"Included Items: {includes}"
        )
        
        metadata = {
            "sku": row.get('sku', ''),
            "name": row.get('name', ''),
            "url": row.get('url', ''),
            "image": row.get('images/0', ''),
            "arm_type": row.get('specs/arm_type', ''),
            "size_max_inch": row.get('specs/size_max_inch', ''),
            "weight_per_arm_kg": row.get('specs/weight_per_arm_kg', ''),
            "vesa": ", ".join(filter(None, [
                row.get('specs/vesa/0', ''), 
                row.get('specs/vesa/1', '')
            ])),
            "desk_thickness_mm": row.get('specs/desk_thickness_mm', ''),
            "compatibility_notes": row.get('compatibility_notes', ''),
            "usb_hub": row.get('specs/usb_hub', ''),
            "includes": includes
        }
        
        metadata_cleaned = {k: v for k, v in metadata.items() if v}

        node_list.append(
            TextNode(
                id_=str(uuid.uuid4()),
                text=doc_content,
                metadata=metadata_cleaned
            )
        )

    add_product_node_batch(nodes=node_list)

if __name__ == "__main__":
    # load_data_and_build_retrievers()
    # seed_products_db()
    pass