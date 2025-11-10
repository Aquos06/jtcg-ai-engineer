import asyncio
import json
import pandas as pd
from uuid import uuid4
from tqdm.asyncio import tqdm 
from typing import List, Dict, Any

from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context
from llama_index.core.llms import ChatMessage, MessageRole

from agent.agent import CRMAgent
from agent.const import JTCG_SYSTEM_PROMPT
from config.env import OPENAI_MODEL

async def _prepare_context(
    agent,
    conversation_history: List[Dict[str, Any]],
    conversation_id: str
) -> Context:
    """
    Creates a new, fresh Context and pre-loads it with the
    system prompt and the conversation history.
    """
    
    context = Context(agent)
    await context.store.set("conversation_id", conversation_id)
    await context.store.set("user_id", None)
    await context.store.set("order_id", None)
    await context.store.set("email", None)
    await context.store.set("waiting_for", None)
    await context.store.set("language", "en")
    
    
    history_chat_messages = [ChatMessage(role=MessageRole.SYSTEM, content=JTCG_SYSTEM_PROMPT)]
    
    for message in conversation_history:
        role = message.get("role")
        try:
            content = message.get("content", [{}])[0].get("text", "")
        except Exception:
            content = ""
            
        if role == "user":
            history_chat_messages.append(
                ChatMessage(role=MessageRole.USER, content=content)
            )
        elif role == "assistant":
            history_chat_messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=content)
            )
            
    await context.store.set("history", history_chat_messages)
    return context

# --- The Main Evaluation Function ---

async def run_evaluation(test_file_path: str, results_file_path: str):
    
    print("Setting up agent and loading data...")
    llm = OpenAI(model=OPENAI_MODEL)
    
    agent = CRMAgent(llm=llm)

    print(f"Loading test cases from {test_file_path}...")
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    results_log = []
    print(f"Starting batch test of {len(test_cases)} questions...")

    for conversation in tqdm(test_cases, desc="Evaluating Agent"):
        agent.tools_called = []
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
            conversation_history=history_messages,
            conversation_id=f"TEST-CONV-{uuid4()}"
        )
        
        # 4. Run the workflow
        final_result = None
        try:
            final_result = await agent.run(input=input_question, ctx=conversation_context)
            
            if final_result is None:
                raise Exception("Workflow did not return a StopEvent")

            if isinstance(final_result, dict):
                response_message = final_result.get("message")
                detected_intent = final_result.get("intent")
                tools_called = str(final_result.get("tools", [])) # Convert list to string for CSV
            else:
                response_message = str(final_result)
                detected_intent = "N/A (String Output)"
                tools_called = "N/A (String Output)"

        except Exception as e:
            print(f"  ERROR on case: {input_question[:30]}... -> {e}")
            response_message = f"WORKFLOW_ERROR: {e}"
            detected_intent = "ERROR"
            tools_called = "ERROR"

        results_log.append({
            "input_history": json.dumps(history_messages, ensure_ascii=False),
            "input_question": input_question,
            "agent_response": response_message,
            "detected_intent": detected_intent,
            "tools_called": tools_called,
        })

    print("\nBatch test complete. Saving to CSV...")
    results_df = pd.DataFrame(results_log)
    
    results_df["is_correct"] = "" 
    
    results_df.to_csv(results_file_path, index=False, encoding='utf-8-sig')
    print(f"Evaluation complete. Results saved to {results_file_path}")

if __name__ == "__main__":
    
    TEST_FILE = "document/evaluation.json" 
    RESULTS_FILE = "output/evaluation_results_final_nano.csv"
    
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