import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from config.env import OPENAI_API_KEY
from tqdm import tqdm

INPUT_CSV_PATH = 'output/merged_responses.csv'  # Your merged file
OUTPUT_CSV_PATH = 'output/evaluation_results_agent_both.csv'
EVALUATOR_MODEL = "gpt-4.1"  # Use gpt-4o or gpt-4-turbo
# ---------------------

# Load the API key from environment variables
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except KeyError:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

# 1. Define the Structured Output using Pydantic
# (REPLACE your models with these new ones)

class ResponseEvaluation(BaseModel):
    """
    An evaluation of a single agent response.
    """
    is_correct: bool = Field(
        ..., 
        description="True if the response is valid and follows all critical rules, False otherwise."
    )
    violations: Optional[List[str]] = Field(
        None, 
        description="A list of Rubric codes (e.g., 'R-1', 'R-3') that this response violated. Null if none."
    )
    score: float = Field(
        ..., 
        ge=0.0, 
        le=5.0, 
        description="A 0.0-5.0 score: 0=Completely wrong, 3=Partially helpful, 5=Perfect."
    )

class EvaluationResult(BaseModel):
    """
    The final comparison and verdict.
    """
    best_response: Literal["response_1", "response_2", "both_correct"] = Field(
        ...,
        description="The final verdict. 'response_1' or 'response_2' if one is clearly better. 'both_correct' if both are valid."
    )
    reasoning: str = Field(
        ...,
        description="A step-by-step comparison explaining the verdict, referencing the Rubric."
    )
    evaluation_1: ResponseEvaluation = Field(..., description="Evaluation details for Agent Response 1.")
    evaluation_2: ResponseEvaluation = Field(..., description="Evaluation details for Agent Response 2.")

EVALUATOR_SYSTEM_PROMPT = """
You are an expert AI Quality Grader. Your task is to compare two agent responses and provide a verdict from three options: `response_1`, `response_2`, or `both_correct`.

# 📜 Quality Checklist (Critical Rules)
A response is **Incorrect** (is_correct: False) if it violates *any* of these.

* **R-1 (Orders):** Did it *fail* to handle an order request correctly? (e.g., didn't ask for `user_id` *before* giving info).
* **R-2 (Transfer):** Did it *fail* to handle a human transfer correctly? (e.g., didn't ask for `email` *before* confirming).
* **R-3 (Honesty):** Is the information inaccurate? Did it *make up* facts, prices, policies, or links?
* **R-4 (Links):** Did it *fail* to provide required links for an FAQ or Product recommendation?
* **R-5 (Intent):** Did it *fail* to answer the user's *most recent question*?
* **R-6 (Redirect):** Did it *fail* to redirect an off-topic question?

# ✅ Stylistic Guidelines (For Scoring)
These rules are not "failures" but should be used to give the 0-5 score.
* **S-1 (Tone):** Is the tone professional, concise, and friendly?
* **S-2 (Next Step):** Does it provide a clear next step?

---

# 📝 TASK

1.  Analyze the user's intent (`input_history` + `input_question`).
2.  **Evaluate Response 1:**
    * Does it violate any **Quality Checklist (R- rules)**?
    * Set `evaluation_1.is_correct` to `True` or `False`.
    * List any violations (e.g., "R-3").
    * Assign `evaluation_1.score` (using S-1 and S-2 as guides).
3.  **Evaluate Response 2:**
    * Does it violate any **Quality Checklist (R- rules)**?
    * Set `evaluation_2.is_correct` to `True` or `False`.
    * List any violations.
    * Assign `evaluation_2.score`.
4.  **Determine the Final `best_response`:**
    * **Case 1:** If R1 `is_correct: True` AND R2 `is_correct: True` -> **`both_correct`**.
    * **Case 2:** If R1 `is_correct: True` AND R2 `is_correct: False` -> **`response_1`**.
    * **Case 3:** If R1 `is_correct: False` AND R2 `is_correct: True` -> **`response_2`**.
    * **Case 4 (Both Incorrect):** If R1 `is_correct: False` AND R2 `is_correct: False` -> You must *force a choice*. Pick the one that is **"less bad"** (e.g., has a higher score, or a less severe violation). Your choice MUST be **`response_1`** or **`response_2`**.
5.  Provide a clear `reasoning` explaining *why* you reached this verdict, referencing the rules.
6.  Respond *only* with the requested structured output.
"""

# 3. The Evaluator Function
def get_evaluation(history: str, question: str, resp1: str, resp2: str) -> Optional[EvaluationResult]:
    """
    Calls the OpenAI API to evaluate the two responses.
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

    ## Agent Response 1
    ```text
    {resp1}
    ```

    ## Agent Response 2
    ```text
    {resp2}
    ```
    """

    try:
        response = client.responses.parse(
            model=EVALUATOR_MODEL,
            input=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            text_format=EvaluationResult,
        )
        return response.output_parsed
    
    except Exception as e:
        print(f"  Error during API call: {e}")
        return None

# 4. Main Script Execution
def main():
    print(f"Loading merged CSV: {INPUT_CSV_PATH}...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {INPUT_CSV_PATH}")
        print("Please run the previous CSV merging script first.")
        return

    # Ensure required columns exist
    required_cols = ['input_history', 'input_question', 'agent_response_1', 'agent_response_2']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input CSV must contain all of the following columns: {required_cols}")
        return

    print(f"Found {len(df)} rows to evaluate.")
    
    evaluation_results = []

    for index, row in tqdm(df.iterrows(), desc="AI Evaluation"):
        print(f"--- Evaluating row {index + 1}/{len(df)} ---")
        
        # Handle potential 'NaN' or 'None' values from CSV
        history = str(row.get('input_history', ''))
        question = str(row.get('input_question', ''))
        resp1 = str(row.get('agent_response_1', ''))
        resp2 = str(row.get('agent_response_2', ''))

        result = get_evaluation(history, question, resp1, resp2)
        
        if result:
            print(f"  Verdict: {result.best_response}")
            evaluation_results.append(result.model_dump())
        else:
            print("  Evaluation failed for this row.")
            # Append empty data to keep row count consistent
            evaluation_results.append({
                "best_response": "ERROR",
                "reasoning": "API call failed.",
                "evaluation_1": None,
                "evaluation_2": None
            })

    print("--- Evaluation complete. ---")

    # Convert the list of Pydantic models (as dicts) into a DataFrame
    # We use pd.json_normalize to "flatten" the nested JSON structures
    results_df = pd.json_normalize(evaluation_results)

    # Combine the original data with the new evaluation data
    final_df = pd.concat([df, results_df], axis=1)
    
    # Save the final, comprehensive report
    final_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    
    print(f"\nSuccessfully saved evaluation results to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()