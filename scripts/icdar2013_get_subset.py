import json
import os

def is_spanning(cells):
    """
    Check if one of the cells is spanning.
    Meaning either the row or the column is spanning: len(row_nums)>1 or len(col_nums)>1.
    """
    for cell in cells:
        row_nums = len(cell["row_nums"])
        col_nums = len(cell["column_nums"])
        if row_nums > 1 or col_nums > 1:
            return True
    return False

def get_simple_tables(data):
    """
    Filter out the tables that do not have spanning cells from the provided data.
    """
    simple_tables = []
    for table in data:
        cells = table['cells']
        if not is_spanning(cells):
            simple_tables.append(table)
    return simple_tables

def process_directory(input_dir, output_dir):
    """
    Process all JSON files in the input directory, filter simple tables, and save the results in the output directory.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            # Read the JSON file
            with open(input_file_path, 'r') as f:
                data = json.load(f)

            # Get simple tables
            simple_tables = get_simple_tables(data)

            # If there are simple tables, save them to the output directory
            if simple_tables:
                with open(output_file_path, 'w') as f:
                    json.dump(simple_tables, f, indent=4)

# Example usage
input_directory = '/path/to/your/input_directory'
output_directory = '/path/to/your/output_directory'
process_directory(input_directory, output_directory)
