import pandas as pd

def analyze_csv_column(file_path, column_name):
    """
    Analyzes a specific column in a CSV file to count TRUE/FALSE values
    and calculate the percentage of TRUE.

    Args:
        file_path (str): The path to the CSV file.
        column_name (str): The name of the column to analyze.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check if the specified column exists
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in the file.")
            print(f"Available columns are: {list(df.columns)}")
            return

        # Get the value counts for the specified column
        # This handles potential string values 'TRUE' and 'FALSE'
        value_counts = df[column_name].value_counts()

        # Extract counts for 'TRUE' and 'FALSE'
        # Using .get() avoids errors if one of the values is missing
        true_count = value_counts.get('TRUE', 0)
        false_count = value_counts.get('FALSE', 0)

        # Handle boolean True/False as well
        if true_count == 0 and false_count == 0:
             true_count = value_counts.get(True, 0)
             false_count = value_counts.get(False, 0)

        total_count = true_count + false_count

        if total_count == 0:
            print(f"No 'TRUE' or 'FALSE' values found in the column '{column_name}'.")
        else:
            # Calculate the percentage of TRUE
            percentage_true = (true_count / total_count) * 100

            # Print the results
            print(f"\n--- Analysis for column: '{column_name}' ---")
            print(f"Total 'TRUE' count: {true_count}")
            print(f"Total 'FALSE' count: {false_count}")
            print(f"Percentage of 'TRUE': {percentage_true:.2f}%")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    FILE_PATH = "output/reliability_results_agent_intent_nano_1.csv"
    COLUMN_NAME="is_correct"
    analyze_csv_column(FILE_PATH, COLUMN_NAME)