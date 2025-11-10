import pandas as pd
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from config.env import OPENAI_API_KEY
from tqdm import tqdm
import time

# --- Configuration ---
# ⚠️ RUN THIS SCRIPT TWICE ⚠️
#
# Run 1:
INPUT_CSV_PATH = 'output/evaluation_results_final_nano.csv'  # 👈 Your ORIGINAL Agent 1 CSV
OUTPUT_CSV_PATH = 'output/reliability_results_agent_intent_nano_1.csv'
#
# Run 2:
# INPUT_CSV_PATH = 'file2.csv' # 👈 Your ORIGINAL Agent 2 CSV
# OUTPUT_CSV_PATH = 'output/reliability_results_agent2.csv'
# ---------------------

EVALUATOR_MODEL = "gpt-4.1"  # "gpt-4.1" isn't a valid model, using "gpt-4o"
AGENT_RESPONSE_COLUMN = 'agent_response' # 👈 Make sure this matches your CSV

# Load the API key from environment variables
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except KeyError:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# 1. Define the NEW, simple Pydantic model
class ReliabilityResult(BaseModel):
    is_correct: bool = Field(
        ..., 
        description="True if the response is valid and follows all critical rules. False otherwise."
    )
    violations: Optional[List[str]] = Field(
        None, 
        description="A list of Rubric codes (e.g., 'R-1', 'R-3') that this response violated. Null if none."
    )

# 2. Define the NEW, reliability-only prompt
EVALUATOR_SYSTEM_PROMPT = """
You are a strict AI Quality Grader. Your task is to evaluate a single agent response for critical rule violations.
You must only determine if the response is 'correct' or 'incorrect'. Do NOT judge for style or tone.

# 📜 Quality Checklist (Critical Rules)
A response is **Incorrect** (is_correct: False) if it violates *any* of these.

* **R-1 (Orders):** Did it *fail* to handle an order request correctly? (e.g., didn't ask for `user_id` *before* giving info).
* **R-2 (Transfer):** Did it *fail* to handle a human transfer correctly? (e.g., didn't ask for `email` *before* confirming).
* **R-3 (Honesty):** Is the information inaccurate? Did it *make up* facts, prices, policies, or links?
* **R-4 (Links):** Did it *fail* to provide required links for an FAQ or Product recommendation?
* **R-5 (Intent):** Did it *fail* to answer the user's *most recent question*?
* **R-6 (Redirect):** Did it *fail* to redirect an off-topic question?

---

# 📝 TASK

1.  Analyze the user's intent (`input_history` + `input_question`).
2.  **Evaluate the `Agent Response`:**
    * Does it violate any **Quality Checklist (R- rules)**?
    * Set `is_correct` to `True` or `False`.
    * List any violations (e.g., "R-3").
3.  Respond *only* with a valid JSON object matching this Pydantic schema:
    {"is_correct": bool, "violations": ["R-1", ...]}
"""

# 3. The Evaluator Function (Using standard OpenAI JSON mode)
def get_reliability_evaluation(history: str, question: str, response: str, intention: str) -> Optional[ReliabilityResult]:
    """
    Calls the OpenAI API to evaluate a single response.
    """
    user_prompt = f"""
    #  Inputs for Evaluation

    ## Chat History
    ```json
    {history}
    ```

    ## User's Current Question
    ```text
    {question}
    ```

    ## Agent Response
    ```text
    {response}
    ```

    ## Agent Intention
    ```text
    {intention}
    ```
    """

    try:
        # Using standard OpenAI client with JSON mode
        response = client.chat.completions.create(
            model=EVALUATOR_MODEL,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        # Parse the JSON string from the response
        result_json = json.loads(response.choices[0].message.content)
        
        # Validate with Pydantic
        return ReliabilityResult(**result_json)
    
    except Exception as e:
        print(f"  Error during API call or JSON parsing: {e}")
        return None

# 4. Main Script Execution
def main():
    print(f"Loading input CSV: {INPUT_CSV_PATH}...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {INPUT_CSV_PATH}")
        return

    # Ensure required columns exist
    required_cols = ['input_history', 'input_question', AGENT_RESPONSE_COLUMN]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input CSV must contain all of the following columns: {required_cols}")
        print(f"Please check the '{AGENT_RESPONSE_COLUMN}' variable in the script.")
        return

    print(f"Found {len(df)} rows to evaluate.")
    
    evaluation_results = []

    # Use tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Reliability Evaluation"):
        
        # Handle potential 'NaN' or 'None' values from CSV
        history = str(row.get('input_history', ''))
        question = str(row.get('input_question', ''))
        intention = str(row.get('detected_intent'))
        resp = str(row.get(AGENT_RESPONSE_COLUMN, ''))

        result = get_reliability_evaluation(history, question, resp,intention)
        
        if result:
            evaluation_results.append(result.model_dump())
        else:
            print(f"  Evaluation failed for row {index + 1}.")
            evaluation_results.append({
                "is_correct": "ERROR",
                "violations": ["API_CALL_FAILED"]
            })

        time.sleep(1)

    print("--- Evaluation complete. ---")

    # Convert the list of Pydantic models (as dicts) into a DataFrame
    results_df = pd.json_normalize(evaluation_results)

    # Combine the original data with the new evaluation data
    final_df = pd.concat([df, results_df], axis=1)
    
    # Save the final, comprehensive report
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    
    print(f"\nSuccessfully saved reliability results to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()