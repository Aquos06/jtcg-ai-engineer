from typing import List
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolOutput, ToolSelection
from llama_index.core.workflow import Event
from llama_index.core.workflow import Event

from agent.schemas import ToolName

class FAQEvent(Event): 
    pass

class ProductEvent(Event):
    pass

class OrderEvent(Event):
    run_tool: ToolName

class HandoverEvent(Event): 
    pass

class GeneralResponseEvent(Event):
    pass

class AskForInfoEvent(Event):
    info_needed: str

class RouterEvent(Event):
    pass

class RejectEvent(Event):
    pass


class InputEvent(Event):
    input: List[ChatMessage]


class ToolCallEvent(Event):
    tool_calls: List[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput


class StreamEvent(Event):
    delta: str
