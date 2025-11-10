import json
import logging

from typing import Any, List, Union
from uuid import uuid4

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from agent.tools import search_knowledge_base, product_search, get_orders_by_user, get_order_details, create_support_ticket
from agent.schemas import ToolName, AgentIntent, UserIntent
from agent.const import JTCG_SYSTEM_PROMPT, ASK_FOR_INFO_PROMPT, INTENT_ROUTER_PROMPT, REJECT_AND_REDIRECT_PROMPT
from agent.event import OrderEvent, ProductEvent, HandoverEvent, AskForInfoEvent, GeneralResponseEvent, FAQEvent, RouterEvent, RejectEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CRMAgent(Workflow):
    def __init__(
        self,
        llm: OpenAI,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.llm = llm

        self.tools = {
            ToolName.SEARCH_KNOWLEDGE_BASE: search_knowledge_base,
            ToolName.PRODUCT_SEARCH: product_search,
            ToolName.GET_ORDER_BY_USER: get_orders_by_user,
            ToolName.GET_ORDER_DETAILS: get_order_details,
            ToolName.CREATE_SUPPORT_TICKET: create_support_ticket
        }
        self.tools_called: List[ToolName] = []
        self.intent = ""

    async def _get_chat_history(self, ctx: Context) -> List[ChatMessage]:
        history:list = await ctx.store.get("history", default=[])
        if not history:
            history.append(ChatMessage(role=MessageRole.SYSTEM, content=JTCG_SYSTEM_PROMPT))
        return history
    
    async def _update_chat_history(self, ctx: Context, message: ChatMessage):
        history = await self._get_chat_history(ctx)
        history.append(message)
        await ctx.store.set("history", history)
    
    async def _synthesize_response(
        self, 
        ctx: Context, 
        tool_name: ToolName, 
        tool_input: dict, 
        tool_output: str
    ) -> str:
        """
        Helper to run the final LLM call to synthesize a human-like answer.
        This version "fakes" the assistant tool call message for better context.
        """
        
        tool_call_id = f"call_{str(uuid4())}"[:30]
        self.tools_called.append(tool_name)
        assistant_tool_call_msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input or {})
                        }
                    }
                ]
            }
        )
        
        tool_output_msg = ChatMessage(
            role=MessageRole.TOOL,
            content=tool_output,
            additional_kwargs={
                "tool_call_id": tool_call_id,
                "name": tool_name
            }
        )
        
        await self._update_chat_history(ctx, assistant_tool_call_msg)
        await self._update_chat_history(ctx, tool_output_msg)
        
        full_history = await self._get_chat_history(ctx)
        
        response = await self.llm.achat(messages=full_history)
        
        await self._update_chat_history(ctx, response.message)
        return response.message.content
    
    @step
    async def get_intent_step(self, ctx: Context, ev: StartEvent) -> RouterEvent:
        """
        Step 1: "Thinking." (The 'get_intent' step)
        """
        user_message_str = ev.input
        await self._update_chat_history(ctx, ChatMessage(role=MessageRole.USER, content=user_message_str))
        
        user_id = await ctx.store.get("user_id", default=None)
        order_id = await ctx.store.get("order_id", default=None)
        email = await ctx.store.get("email", default=None)
        waiting_for = await ctx.store.get("waiting_for", default=None)
        
        chat_history = await self._get_chat_history(ctx)
        
        prompt = INTENT_ROUTER_PROMPT.format(user_id=user_id, order_id=order_id, email=email, waiting_for=waiting_for)
        
        strucured_llm = self.llm.as_structured_llm(
            AgentIntent,
        )
        messages= [ChatMessage(role=MessageRole.SYSTEM, content=prompt)] + chat_history
        response = strucured_llm.chat(messages=messages)
        
        await ctx.store.set("intent_plan", response)

        response_raw = response.raw
        return RouterEvent(input=response_raw)

    @step
    async def router_step(self, ctx: Context, ev: RouterEvent) -> Union[AskForInfoEvent, OrderEvent, FAQEvent, AskForInfoEvent, ProductEvent, HandoverEvent, GeneralResponseEvent, RejectEvent]:
        """
        Step 2: "Python Logic Router."
        """
        plan: AgentIntent = ev.input
        await ctx.store.set("language", plan.language) 

        if plan.entities.user_id:
            await ctx.store.set("user_id", plan.entities.user_id)
        if plan.entities.order_id:
            await ctx.store.set("order_id", plan.entities.order_id)
        if plan.entities.email:
            await ctx.store.set("email", plan.entities.email)
        
        self.intent = plan.intent
        if self.intent == UserIntent.REJECT_REQUEST:
            logger.info("Routing to Reject Request Worker.")
            return RejectEvent(input=plan)
        
        if self.intent == UserIntent.ORDER_INFO:
            user_id = await ctx.store.get("user_id", default=None)
            order_id = await ctx.store.get("order_id", default=None)
            
            if not user_id:
                await ctx.store.set("waiting_for", "user_id")
                return AskForInfoEvent(info_needed="user_id")
            if not order_id:
                return OrderEvent(run_tool=ToolName.GET_ORDER_BY_USER)
            
            return OrderEvent(run_tool=ToolName.GET_ORDER_DETAILS)
            
        if self.intent == UserIntent.PRODUCT_SEARCH:
            return ProductEvent(input=plan) 
            
        if self.intent == UserIntent.FAQ:
            return FAQEvent(input=plan)
            
        if self.intent == UserIntent.HUMAN_HANDOVER:
            email = await ctx.store.get("email", default=None)
            if not email:
                await ctx.store.set("waiting_for", "email")
                return AskForInfoEvent(info_needed="email")
            return HandoverEvent(input=plan)

        return GeneralResponseEvent(input=plan)
    
    @step
    async def reject_request_worker_step(self, ctx: Context, ev: RejectEvent) -> StopEvent:
        """
        Handles out-of-scope requests by politely declining and
        redirecting the user back to the agent's capabilities.
        """
        plan: AgentIntent = ev.input
        language = await ctx.store.get("language", default="en")
        
        logger.info("Running Reject Request Worker...")

        prompt = REJECT_AND_REDIRECT_PROMPT.format(language=language)
        
        chat_history = await self._get_chat_history(ctx)
        
        response = await self.llm.achat(
            messages=[ChatMessage(role=MessageRole.SYSTEM, content=prompt)] + chat_history[1:]
        )
        
        await self._update_chat_history(ctx, response.message)
        
        tools_called = await ctx.store.get("tools_called", default=[])
        
        return StopEvent(result={
            "message": response.message.content,
            "intent": plan.intent,
            "tools": tools_called
        })
    
    @step
    async def ask_for_info_worker_step(self, ctx: Context, ev: AskForInfoEvent) -> StopEvent:
        """
        Worker step that ONLY generates human-like questions.
        """
        info_needed = ev.info_needed
        language = await ctx.store.get("language", default="en")
        
        prompt = ASK_FOR_INFO_PROMPT.format(
            language=language, 
            info_needed=info_needed,
            context_message=""
        )
        
        response = await self.llm.achat(messages=[ChatMessage(role=MessageRole.SYSTEM, content=prompt)])
        await self._update_chat_history(ctx, response.message)
        return self.return_event(response.message.content)
    
    @step
    async def product_worker_step(self, ctx: Context, ev: ProductEvent) -> StopEvent:
        """
        Worker step that handles product search using a "Permissive" strategy.
        
        - If the user provides *any* search criteria (query, size, etc.), 
          it runs the search immediately.
        - If the user provides *no* criteria (e.g., "I want a product"), 
          it asks for more information.
        """
        plan: AgentIntent = ev.input
        tool = self.tools[ToolName.PRODUCT_SEARCH]
        
        searchable_entities = {
            "query": plan.entities.product_query,
            "size_inch": plan.entities.size_inch,
            "weight_kg": plan.entities.weight_kg,
            "arm_type": plan.entities.arm_type,
            "vesa": plan.entities.vesa,
            "desk_thickness_mm": plan.entities.desk_thickness_mm
        }
        
        tool_input_cleaned = {k: v for k, v in searchable_entities.items() if v is not None}
        
        logger.info(f"Product worker: Entities found. Running search with {tool_input_cleaned}")

        tool_output_dict = tool(**tool_input_cleaned)
        tool_output_str = json.dumps(tool_output_dict)

        response_str = await self._synthesize_response(
            ctx,
            tool_name=ToolName.PRODUCT_SEARCH,
            tool_input=tool_input_cleaned,
            tool_output=tool_output_str
        )
        
        tools_called = await ctx.store.get("tools_called", default=[])
        return StopEvent(result={
            "message": response_str,
            "intent": plan.intent,
            "tools": tools_called
        })
    @step
    async def faq_worker_step(self, ctx: Context, ev: FAQEvent) -> StopEvent:
        """Worker step that ONLY handles FAQs."""
        plan: AgentIntent = ev.input
        tool = self.tools[ToolName.SEARCH_KNOWLEDGE_BASE]
        
        logger.info(f"Running FAQ Worker for: {plan.summary_for_next_step}")
        tool_input = {"query": plan.summary_for_next_step}
        
        logger.info(f"Running FAQ Worker for: {plan.summary_for_next_step}")
        tool_output = tool(**tool_input)
        
        response_str = await self._synthesize_response(
            ctx,
            tool_name=ToolName.SEARCH_KNOWLEDGE_BASE,
            tool_input=tool_input,
            tool_output=str(tool_output)
        )
        return self.return_event(response_str)

    @step
    async def order_worker_step(self, ctx: Context, ev: OrderEvent) -> StopEvent:
        """Worker step that handles both order tool calls."""
        run_tool = ev.run_tool
        user_id = await ctx.store.get("user_id")
        
        if run_tool == ToolName.GET_ORDER_BY_USER:
            logger.info("Running Order Worker: get_orders_by_user")
            user_id = await ctx.store.get("user_id")
            tool_input = {"user_id": user_id}
            tool_output = self.tools[ToolName.GET_ORDER_BY_USER](**tool_input)
            
            response_str = await self._synthesize_response(
                ctx,
                tool_name=ToolName.GET_ORDER_BY_USER,
                tool_input=tool_input,
                tool_output=str(tool_output)
            )
            return self.return_event(response_str)
        elif run_tool == ToolName.GET_ORDER_DETAILS:
            logger.info("Running Order Worker: get_order_details")
            order_id = await ctx.store.get("order_id")
            
            tool_input = {"order_id": order_id, "user_id": user_id}
            tool_output = self.tools[ToolName.GET_ORDER_DETAILS](**tool_input)
            
            response_str = await self._synthesize_response(
                ctx,
                tool_name=ToolName.GET_ORDER_DETAILS,
                tool_input=tool_input,
                tool_output=str(tool_output)
            )
            return self.return_event(response_str)
            
        return self.return_event("Error in order workflow.")

    @step
    async def handover_worker_step(self, ctx: Context, ev: HandoverEvent) -> StopEvent:
        """
        Worker step that performs the human handover.
        1. Generates a summary.
        2. Calls the (synchronous) handover_simple function.
        """
        logger.info("Running Handover Worker...")
        
        email = await ctx.store.get("email")
        conversation_id = await ctx.store.get("conversation_id")
        chat_history = await self._get_chat_history(ctx)
        
        summary_prompt = "Summarize this chat history for a human support agent. Be concise."
        summary_response = await self.llm.achat(
            messages=chat_history + [ChatMessage(role=MessageRole.SYSTEM, content=summary_prompt)]
        )
        summary = summary_response.message.content
        
        logger.info(f"Calling handover_simple for conv_id {conversation_id}")
        result_string = create_support_ticket(
            conversation_id=conversation_id,
            email=email,
            summary=summary
        )
        self.tools_called.append(ToolName.CREATE_SUPPORT_TICKET)
        
        await self._update_chat_history(ctx, ChatMessage(role=MessageRole.ASSISTANT, content=result_string))
        
        await ctx.store.set("email", None)
        await ctx.store.set("waiting_for", None)
        
        return self.return_event(result_string)

    @step
    async def general_response_worker_step(self, ctx: Context, ev: GeneralResponseEvent) -> StopEvent:
        """Handles greetings, off-topic, etc. No tools."""
        logger.info("Running General Response Worker...")
        chat_history = await self._get_chat_history(ctx)
        chat_history.append(ChatMessage(role=MessageRole.SYSTEM, content="Politely respond to the user's last message."))
        
        response = await self.llm.achat(messages=chat_history)
        await self._update_chat_history(ctx, response.message)
        return self.return_event(response.message.content)
    
    def return_event(self, result: str) -> None:
        return StopEvent(result={
            "message": result,
            "intent": self.intent,
            "tools": self.tools_called
        })