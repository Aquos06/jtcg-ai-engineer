import asyncio
import json
import pandas as pd
from uuid import uuid4
from tqdm.asyncio import tqdm
from typing import List, Dict, Any

from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

# Import your new agent
from agent.agent_auto import CRMAutoAgent 
from config.env import OPENAI_MODEL

async def _prepare_context(
    agent,
    llm: OpenAI,
    conversation_history: List[Dict[str, Any]],
) -> Context:
    """
    Creates a new, fresh Context and pre-loads it with a
    ChatMemoryBuffer containing the conversation history,
    which the CRMAutoAgent workflow expects.
    """
    
    context = Context(agent) 
    
    # Initialize the memory buffer the agent will use
    memory = ChatMemoryBuffer.from_defaults(llm=llm)
    
    # Pre-load the memory with the conversation history
    for message in conversation_history:
        role_str = message.get("role")
        try:
            content = message.get("content", [{}])[0].get("text", "")
        except Exception:
            content = ""
            
        # Determine the correct MessageRole
        role = MessageRole.USER if role_str == "user" else MessageRole.ASSISTANT
            
        if content: # Add the message to the buffer
            memory.put(ChatMessage(role=role, content=content))
            
    await context.store.set("memory", memory)
    return context


async def run_evaluation(test_file_path: str, results_file_path: str):
    
    print("Setting up LLM and loading data...")
    llm = OpenAI(model=OPENAI_MODEL)
    

    print(f"Loading test cases from {test_file_path}...")
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    results_log = []
    print(f"Starting batch test of {len(test_cases)} questions...")

    for conversation in tqdm(test_cases, desc="Evaluating Agent"):
        
        agent = CRMAutoAgent(
            llm=llm, 
            conversation_id=uuid4()
        )
        
        history_messages = conversation[:-1]
        final_query_message = conversation[-1]
        
        try:
            input_question = final_query_message.get("content", [{}])[0].get("text", "")
            if final_query_message.get("role") != "user":
                raise ValueError("Last message in test case is not from user.")
        except Exception as e:
            print(f"Skipping malformed test case: {e}")
            continue

        conversation_context = await _prepare_context(
            agent=agent,
            llm=llm,
            conversation_history=history_messages,
        )
        
        # 4. Run the workflow
        final_result = None
        try:
            final_result = await agent.run(
                input=input_question, 
                ctx=conversation_context
            )
            
            if final_result is None:
                raise Exception("Workflow did not return a StopEvent")

            if isinstance(final_result, dict):
                response_message = final_result.get("response")
            else:
                response_message = str(final_result)

        except Exception as e:
            print(f"  ERROR on case: {input_question[:30]}... -> {e}")
            response_message = f"WORKFLOW_ERROR: {e}"

        results_log.append({
            "input_history": json.dumps(history_messages, ensure_ascii=False),
            "input_question": input_question,
            "agent_response": response_message,
            "is_correct": ""
        })

    print("\nBatch test complete. Saving to CSV...")
    results_df = pd.DataFrame(results_log)
    
    results_df["is_correct"] = "" 
    
    results_df.to_csv(results_file_path, index=False, encoding='utf-8-sig')
    print(f"Evaluation complete. Results saved to {results_file_path}")

if __name__ == "__main__":
    
    TEST_FILE = "document/evaluation.json" 
    RESULTS_FILE = "output/evaluation_results_final_autoagen_nano.csv" # Changed output file
    
    try:
        asyncio.run(run_evaluation(
            test_file_path=TEST_FILE, 
            results_file_path=RESULTS_FILE
        ))
            
    except FileNotFoundError:
        print(f"Error: Test file not found at '{TEST_FILE}'")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")