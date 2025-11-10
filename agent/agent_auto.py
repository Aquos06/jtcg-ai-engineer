import logging
from typing import Any, List
from pydantic import UUID4

from llama_index.core.tools.types import BaseTool
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer

from agent.const import PRODUCT_SEARCH_DESC, GET_ORDER_DETAIL_DESC, GET_ORDER_BY_USER_DESC, CREATE_SUPPORT_TICKET_DESC, SEARCH_KNOWLEDGE_BASE_DESC
from agent.tools import search_knowledge_base, product_search, get_orders_by_user, get_order_details, create_support_ticket
from agent.schemas import ToolName
from agent.event import InputEvent, ToolCallEvent, StreamEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CRMAutoAgent(Workflow):
    def __init__(
        self,
        llm: OpenAI,
        conversation_id: UUID4,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.conversation_id = conversation_id
        self.tools_called: List[ToolName] = []

    def create_support_ticket(self, email: str, summary: str):
        return create_support_ticket(
            conversation_id=self.conversation_id,
            email=email,
            summary=summary
        )

    @property
    def tools(self) -> List[BaseTool]:
        search_knowledge_base_tool = FunctionTool.from_defaults(
            fn=search_knowledge_base,
            name=ToolName.SEARCH_KNOWLEDGE_BASE,
            description=SEARCH_KNOWLEDGE_BASE_DESC
        )
        product_search_tool = FunctionTool.from_defaults(
            fn=product_search,
            name=ToolName.PRODUCT_SEARCH,
            description=PRODUCT_SEARCH_DESC
        )
        get_orders_by_user_tool = FunctionTool.from_defaults(
            fn=get_orders_by_user,
            name=ToolName.GET_ORDER_BY_USER,
            description=GET_ORDER_BY_USER_DESC
        )
        get_order_detail_tool = FunctionTool.from_defaults(
            fn=get_order_details,
            name=ToolName.GET_ORDER_DETAILS,
            description=GET_ORDER_DETAIL_DESC
        )
        create_support_ticket_tool = FunctionTool.from_defaults(
            fn=self.create_support_ticket,
            name=ToolName.CREATE_SUPPORT_TICKET,
            description=CREATE_SUPPORT_TICKET_DESC
        )

        return [
            search_knowledge_base_tool,
            product_search_tool,
            get_orders_by_user_tool,
            get_order_detail_tool,
            create_support_ticket_tool
        ]
    
    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: StartEvent
    ) -> InputEvent:
        # clear sources
        await ctx.store.set("sources", [])

        # check if memory is setup
        memory = await ctx.store.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # get chat history
        chat_history = memory.get()

        # update context
        await ctx.store.set("memory", memory)

        return InputEvent(input=chat_history)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input

        response_stream = await self.llm.astream_chat_with_tools(
            self.tools, chat_history=chat_history
        )
        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        memory = await ctx.store.get("memory")
        memory.put(response.message)
        await ctx.store.set("memory", memory)

        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            sources = await ctx.store.get("sources", default=[])
            return StopEvent(
                result={"response": response.message.content, "sources": [*sources]}
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> InputEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []
        sources = await ctx.store.get("sources", default=[])

        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        memory = await ctx.store.get("memory")
        for msg in tool_msgs:
            memory.put(msg)

        await ctx.store.set("sources", sources)
        await ctx.store.set("memory", memory)

        chat_history = memory.get()
        return InputEvent(input=chat_history)
   