import pandas as pd

# --- Configuration ---
FILE_PATH = 'output/evaluation_results_agent_both.csv'  # <--- !! IMPORTANT: Change this to your CSV file's path
COLUMN_TO_ANALYZE = 'best_response'
# ---

def analyze_best_response(file_path, column_name):
    """
    Calculates the total counts and percentages for the 'best_response' column.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check if the required column exists
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in the CSV.")
            print(f"Available columns are: {list(df.columns)}")
            return

        # 1. Calculate the total counts for each unique value
        print("--- Total Counts ---")
        counts = df[column_name].value_counts()
        print(counts)
        print("\n" + "="*30 + "\n")

        # 2. Calculate the percentages
        print("--- Percentages ---")
        # normalize=True calculates the frequency (proportion)
        percentages = df[column_name].value_counts(normalize=True) * 100
        
        # Format the percentages to 2 decimal places
        formatted_percentages = percentages.map('{:.2f}%'.format)
        print(formatted_percentages)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Run the analysis ---
if __name__ == "__main__":
    analyze_best_response(FILE_PATH, COLUMN_TO_ANALYZE)