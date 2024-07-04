import os
import pandas as pd
import glob

# Define the file pattern
file_pattern = 'C:/Users/kiran/OneDrive/Desktop/Internship/data/HP*.txt'
file_paths = glob.glob(file_pattern)

for file_path in file_paths:
    # Read the file into a DataFrame
    df = pd.read_csv(file_path, delimiter=';')  # Assuming the delimiter is tab, adjust if necessary

    # Drop the 'Alpha' column if it exists
    if 'G_alpha' in df.columns:
        df.drop(columns=['G_alpha'], inplace=True)
    else : print("x")
    # Save the updated DataFrame to a temporary file
    temp_file_path = file_path + '.tmp'
    df.to_csv(temp_file_path, index=False, sep=';')  # Keeping the delimiter as tab, adjust if necessary
    
    # Replace the original file with the temporary file
    os.replace(temp_file_path, file_path)
    
    print(f"Updated file: {file_path}")
