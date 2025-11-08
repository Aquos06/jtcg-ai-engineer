import logging
import uuid
import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context, StopEvent

from agent.agent import CRMAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

async def main():
    """
    Sets up and runs the JTCG Agent Workflow in a chat loop.
    """
    
    try:
        llm = OpenAI(model="gpt-4.1")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI LLM: {e}")
        logger.error("Please make sure your OPENAI_API_KEY environment variable is set.")
        return

    conversation_id = f"JTCG-CHAT-{uuid.uuid4()}"
    agent = CRMAgent(
        llm=llm,
    )
    
    # --- Chat Loop ---
    print("--- JTCG 'Senior Skill' Agent Workflow Initialized ---")
    print("This version uses a 'get_intent' -> 'router' -> 'worker' graph.")
    print("Type 'exit' or 'quit' to end the chat.")
    
    context = Context(agent)
    await context.store.set("conversation_id", conversation_id)
    await context.store.set("history", [])
    await context.store.set("user_id", None)
    await context.store.set("order_id", None)
    await context.store.set("email", None)
    await context.store.set("waiting_for", None)
    await context.store.set("language", "en")
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Agent: Goodbye!")
                break
                
            result = await agent.run(input=user_input, ctx=context)
            print(f"\nAgent: {result}")


        except KeyboardInterrupt:
            print("\nAgent: Goodbye!")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())