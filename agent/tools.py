import re
from pydantic import UUID4
from typing import List, Union, Dict, Optional, Any
from retriever.utils import retreive_from_vector_store, retrieve_from_product
from document.data import product_df, order_db

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

def search_knowledge_base(query: str) -> Dict[str, Union[str, List[str]]]:
    """Searches the knowledge base using a hybrid approach."""
    vector_results = retreive_from_vector_store(query)
    return {"status": "success", "results": [node.get_text() for node in vector_results]}

def product_search(query: Optional[str] = None, size_inch: Optional[int] = None, weight_kg: Optional[float] = None) -> Dict[str, Any]:
    """Searches the product catalog based on specifications."""
    filtered_df = product_df.copy()
    if query:
        filtered_df = filtered_df[filtered_df['name'].str.contains(query, case=False)]
    if size_inch:
        filtered_df = filtered_df[filtered_df['specs/size_max_inch'] >= size_inch]
    if weight_kg:
        filtered_df = filtered_df[
            (filtered_df['weight_min_kg'] <= weight_kg) &
            (filtered_df['weight_max_kg'] >= weight_kg)
        ]
    results = filtered_df.to_dict('records')
    product_list = [{ "sku": r["sku"], "name": r["name"], "url": r["url"], "image": r["images/0"], "compatibility": r["compatibility_notes"]} for r in results]
    return {"status": "success", "products": product_list}

def product_search(
    query: Optional[str] = None,
    size_inch: Optional[int] = None,
    weight_kg: Optional[float] = None,
    arm_type: Optional[str] = None,
    vesa: Optional[str] = None,
    desk_thickness_mm: Optional[int] = None
) -> Dict[str, Any]:
    """
    Searches the product catalog.
    - query: Searches name, compatibility notes, and included items.
    - size_inch: Finds arms that support *at least* this size.
    - weight_kg: Finds arms where the weight is *within* the supported range.
    - arm_type: Filters by type (e.g., 'wall_mount', 'accessory', 'dual_gas_spring').
    - vesa: Checks if the arm supports a VESA standard (e.g., '75x75', '100x100').
    - desk_thickness_mm: Checks if the desk thickness is within the supported range.
    """
    filtered_df = product_df.copy()

    result_query = []
    if query:
        result = retrieve_from_product(query)
        result_query = [n.metadata for n in result]

    if arm_type:
        filtered_df = filtered_df[filtered_df['specs/arm_type'] == arm_type]

    if size_inch:
        filtered_df = filtered_df[
            (filtered_df['specs/size_max_inch'] >= size_inch) | 
            (filtered_df['specs/arm_type'] == 'accessory')
        ]

    if weight_kg:
        filtered_df = filtered_df[
            (filtered_df['weight_min_kg'] <= weight_kg) &
            (filtered_df['weight_max_kg'] >= weight_kg)
        ]

    if vesa:
        filtered_df = filtered_df[
            (filtered_df['specs/vesa/0'] == vesa) |
            (filtered_df['specs/vesa/1'] == vesa)
        ]
        
    if desk_thickness_mm:
        filtered_df = filtered_df[
            (filtered_df['desk_min_mm'] <= desk_thickness_mm) &
            (filtered_df['desk_max_mm'] >= desk_thickness_mm)
        ]

    results = filtered_df.to_dict('records')
    if not query and not arm_type and not size_inch and not weight_kg and not vesa and not desk_thickness_mm:
        results = product_df.to_dict("records")

    product_list = [{
        "sku": r["sku"],
        "name": r["name"],
        "url": r["url"],
        "image": r["images/0"],
        "compatibility": r["compatibility_notes"]
    } for r in results]

    product_list.extend(result_query)

    return {"status": "success", "products": product_list}

def get_orders_by_user(user_id: str) -> Dict[str, Any]:
    """Gets a summary list of orders for a user_id."""
    if user_id not in order_db:
        return {"status": "not_found", "message": "User ID not found."}
    user_orders = order_db.get(user_id, {}).get("orders", [])
    if not user_orders:
        return {"status": "no_orders", "message": "This user has no orders."}
    summaries = [{"order_id": o["order_id"], "placed_at": o["placed_at"], "summary": o["items"][0]["name"]} for o in user_orders]
    return {"status": "success", "orders": summaries}

def get_order_details(order_id: str, user_id: str) -> Dict[str, Any]:
    """Gets full details for a single order_id."""
    if user_id not in order_db:
        return {"status": "not_found", "message": "User ID not found."}

    user_orders = order_db.get(user_id, {}).get("orders", [])

    for order in user_orders:
        if order["order_id"] == order_id:
            return {"status": "success", "details": order}
    return {"status": "not_found", "message": "Order ID not found."}

def create_support_ticket(conversation_id: UUID4, email: str, summary: str) -> str:
    """
    簡易轉接真人（模擬版）
    Args:
        conversation_id: 模擬的對話 ID
        email: 使用者 Email（此版僅做基本格式檢查）
        summary: 對話摘要（會送入 mock API；此處不做實際保存）
        simulate_fail: 設為 True 可強制模擬 API 失敗
    Returns:
        str: 成功 -> "已為您轉接真人"
             失敗 -> "轉接真人時發生錯誤，請聯繫技術團隊協助"
    """
    # 基本 Email 檢查（不通過就視為失敗）
    if not EMAIL_RE.match(email or ""):
        return "轉接真人時發生錯誤，請聯繫技術團隊協助"

    payload = {
        "conversation_id": str(conversation_id),
        "email": email,
        "summary": (summary or "")[:500],  # 簡單截斷，避免過長
    }

    try:
        ok = _mock_api_call(payload, simulate_fail=False)
        return "已為您轉接真人" if ok else "轉接真人時發生錯誤，請聯繫技術團隊協助"
    except Exception:
        return "轉接真人時發生錯誤，請聯繫技術團隊協助"
    
def _mock_api_call(payload: dict, simulate_fail: bool = False) -> bool:
    """
    假的 API 呼叫：預設成功。
    """
    if simulate_fail:
        return False
    if payload.get("conversation_id", "").upper().startswith("FAIL"):
        return False
    return True