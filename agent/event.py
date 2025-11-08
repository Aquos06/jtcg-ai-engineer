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