import json
import pandas as pd
from pandas.core.frame import DataFrame

def get_order_db() -> dict:
    with open("document/order.json", "r", encoding="utf-8") as f:
        orders_db = json.load(f)["orders_db"]
    return orders_db

def get_product_df() -> DataFrame:
    products_df = pd.read_csv("document/product.csv")
    try:
        weight_split = products_df['specs/weight_per_arm_kg'].str.split('-', expand=True).astype(float)
        products_df['weight_min_kg'] = weight_split[0]
        products_df['weight_max_kg'] = weight_split[1]
    except Exception:
        products_df['weight_min_kg'] = 0.0
        products_df['weight_max_kg'] = 99.0

    return products_df

order_db = get_order_db()
product_df = get_product_df()