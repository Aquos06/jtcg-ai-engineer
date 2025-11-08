from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ToolName(str, Enum):
    SEARCH_KNOWLEDGE_BASE="search_knowldge_base"
    PRODUCT_SEARCH="product_search"
    GET_ORDER_BY_USER="get_order_by_user"
    GET_ORDER_DETAILS="get_order_details"
    CREATE_SUPPORT_TICKET="create_support_ticket"

class UserIntent(str, Enum):
    ORDER_STATUS="order_status"
    PRODUCT_SEARCH="product_search"
    FAQ="faq"
    HUMAN_HANDOVER="human_handover"
    GENERAL_RESPONSE="general_response"
    PROVIDING_INFO="providing_info"

class AgentIntent(BaseModel):
    """
    The agent's plan. It classifies the user's intent, detects the
    language, extracts entities, and provides a summary for the next step.
    """
    intent: UserIntent = Field(description="The user's primary goal. Must be one of: 'order_status', 'product_search', 'faq', 'human_handover', 'general_response', 'providing_info'.")
    language: Optional[str] = Field(default="en", description="The language the user is speaking (e.g., 'Traditional Chinese', 'English').")
    entities: Optional[Dict[str, str]] = Field(default=dict, description="Any extracted entities, e.g., {'user_id': 'u_123456', 'order_id': 'JTCG-10001', 'email': 'user@example.com'}")
    summary_for_next_step: str = Field(description="A concise summary of the user's request for the next tool.")