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
    ORDER_INFO="order_info"
    PRODUCT_SEARCH="product_search"
    FAQ="faq"
    HUMAN_HANDOVER="human_handover"
    GENERAL_RESPONSE="general_response"
    REJECT_REQUEST = "reject_request" 

class ExtractedEntities(BaseModel):
    """A structured object for all extracted entities. All fields are optional."""
    
    # --- Workflow Entities ---
    user_id: Optional[str] = Field(None, description="The user's unique ID, e.g., 'u_123456'")
    order_id: Optional[str] = Field(None, description="The order ID, e.g., 'JTCG-202508-10001'")
    email: Optional[str] = Field(None, description="The user's email address")
    
    # --- Product Search Entities ---
    product_query: Optional[str] = Field(None, description="What user ask")
    size_inch: Optional[int] = Field(None, description="Monitor size in inches")
    weight_kg: Optional[float] = Field(None, description="Monitor weight in kg")
    arm_type: Optional[str] = Field(None, description="e.g., 'wall_mount', 'dual_gas_spring'")
    vesa: Optional[str] = Field(None, description="e.g., '75x75', '100x100'")
    desk_thickness_mm: Optional[int] = Field(None, description="User's desk thickness")

class AgentIntent(BaseModel):
    """
    The agent's plan. It classifies the user's intent, detects the
    language, extracts entities, and provides a summary for the next step.
    """
    intent: UserIntent = Field(description="The user's primary goal. Must be one of: 'order_info', 'product_search', 'faq', 'human_handover', 'general_response'")
    language: Optional[str] = Field(default="en", description="The language the user is speaking (e.g., 'Traditional Chinese', 'English').")
    entities: ExtractedEntities = Field(description="Any extracted entities, e.g., {'user_id': 'u_123456', 'order_id': 'JTCG-10001', 'email': 'user@example.com'}")
    summary_for_next_step: str = Field(description="A concise summary of the user's request for the next tool.")