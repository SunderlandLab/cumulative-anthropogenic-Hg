import sys
import os
import pandas as pd
import argparse

def convert_tab_to_csv(input_filepath, remove_tab_file=False):
    """
    Converts a .tab file to a .csv file using pandas.
    
    Args:
        input_filepath (str): Path to the input .tab file.
        remove_tab_file (bool): Whether to remove the original .tab file after conversion.
    """
    # Ensure the file exists
    if not os.path.isfile(input_filepath):
        print(f"Error: File not found - {input_filepath}")
        sys.exit(1)

    # Ensure the file is not empty
    if os.stat(input_filepath).st_size == 0:
        print(f"Error: Input file {input_filepath} is empty.")
        sys.exit(1)

    # Construct output file path (replace .tab with .csv)
    output_filepath = input_filepath.replace('.tab', '.csv')

    try:
        # Read the .tab file using pandas
        df = pd.read_csv(input_filepath, sep='\t')

        # Save as .csv
        df.to_csv(output_filepath, index=False)

        print(f"Conversion successful: {output_filepath}")
    except pd.errors.EmptyDataError:
        print(f"Error: Input file {input_filepath} is empty or malformed.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: Failed to parse {input_filepath}. Ensure it is a valid tab-separated file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

    # Optionally remove the .tab file
    if remove_tab_file:
        try:
            os.remove(input_filepath)
            print(f"Removed file: {input_filepath}")
        except Exception as e:
            print(f"Error removing file {input_filepath}: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert .tab files to .csv")
    parser.add_argument("input_file", help="Path to the .tab file")
    parser.add_argument("--remove", action="store_true", help="Remove the original .tab file after conversion")
    args = parser.parse_args()

    convert_tab_to_csv(args.input_file, remove_tab_file=args.remove)

