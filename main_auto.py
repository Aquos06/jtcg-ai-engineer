import logging
import uuid
import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context

from config.env import OPENAI_MODEL
from agent.event import StreamEvent
from agent.agent_auto import CRMAutoAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

async def main():
    """
    Sets up and runs the JTCG Agent Workflow in a chat loop.
    """
    
    try:
        llm = OpenAI(model=OPENAI_MODEL)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI LLM: {e}")
        logger.error("Please make sure your OPENAI_API_KEY environment variable is set.")
        return

    conversation_id = f"JTCG-CHAT-{uuid.uuid4()}"
    agent = CRMAutoAgent(
        llm=llm,
        conversation_id=conversation_id
    )
    
    # --- Chat Loop ---
    print("--- JTCG 'Senior Skill' Agent Workflow Initialized ---")
    print("This version uses a 'get_intent' -> 'router' -> 'worker' graph.")
    print("Type 'exit' or 'quit' to end the chat.")
    
    context = Context(agent)
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Agent: Goodbye!")
                break
                
            handler = await agent.run(input=user_input, ctx=context)
            print(f"Agent: {handler['response']}")

        except KeyboardInterrupt:
            print("\nAgent: Goodbye!")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())