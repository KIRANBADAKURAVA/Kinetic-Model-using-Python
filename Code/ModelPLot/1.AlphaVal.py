import pandas as pd
import numpy as np
import glob
import os

# Define the file pattern and get the list of files
file_pattern = r"C:\Users\kiran\OneDrive\Desktop\Internship\data\HP*.txt"
file_paths = glob.glob(file_pattern)

# Function to process each file
def process_file(file_path):
    filename = os.path.basename(file_path).split('.')[0]
    conditions = filename.split('_')
    condition_1 = conditions[0][2:]
    condition_2 = conditions[1]
    condition = f'{condition_1}_{condition_2}'
    
    try:
        df = pd.read_csv(file_path, sep=';', header=0, encoding='latin1')
        df["Alpha"] = (100 - df["Mass/%"]) / (100 - df["Mass/%"].iloc[-1])
        data_frames[condition] = df
    
    except pd.errors.ParserError as e:
        print(f"ParserError in file {file_path}: {e}")
        return None
    
    return condition

# Function to apply FAlpha model
def FAlpha(data_frames, condition, model):
    alpha = data_frames[condition]["Alpha"]
    
    if model == 'D1':
        data_frames[condition]["F_alpha"] = 1 / (2 * alpha)
        data_frames[condition]["G_alpha"] = alpha ** 2
    
    elif model == 'D2':
        data_frames[condition]["F_alpha"] = (-np.log(1 - alpha)) ** (-1)
        data_frames[condition]["G_alpha"] = (1 - alpha) * np.log(1 - alpha) + alpha
    
    elif model == 'D3':
        part1 = ((3 / 2) * (1 - alpha)) ** (2 / 3)
        part2 = 1 - (1 - alpha) ** (1 / 3)
        data_frames[condition]["F_alpha"] = part1 * part2
        data_frames[condition]["G_alpha"] = (1 - (1 - alpha) ** (1 / 3)) ** 2
    
    elif model == 'D4':
        data_frames[condition]["F_alpha"] = (((1 - alpha) ** (-1 / 3) - 1) ** (-1 / 3)) * (3 / 2)
        term1 = 1 - (2 * alpha / 3)
        term2 = (1 - alpha) ** (2 / 3)
        data_frames[condition]["G_alpha"] = term1 - term2
    
    elif model == 'F0':
        data_frames[condition]["F_alpha"] = 1
        data_frames[condition]["G_alpha"] = alpha
    
    elif model == 'F1':
        data_frames[condition]["F_alpha"] = 1 - alpha
        data_frames[condition]["G_alpha"] = -np.log(1 - alpha)
    
    elif model == 'R2':
        data_frames[condition]["F_alpha"] = 2 * ((1 - alpha) ** (1 / 2))
        data_frames[condition]["G_alpha"] = 1 - ((1 - alpha) ** (1 / 2))
    
    elif model == 'R3':
        data_frames[condition]["F_alpha"] = 3 * ((1 - alpha) ** (2 / 3))
        data_frames[condition]["G_alpha"] = 1 - ((1 - alpha) ** (1 / 3))
    
    elif model == 'A2':
        data_frames[condition]["F_alpha"] = 2 * (1 - alpha) * (-np.log(1 - alpha)) ** (1 / 2)
        data_frames[condition]["G_alpha"] = (-np.log(1 - alpha)) ** (1 / 2)
    
    elif model == 'A3':
        data_frames[condition]["F_alpha"] = 3 * (1 - alpha) * (-np.log(1 - alpha)) ** (2 / 3)
        data_frames[condition]["G_alpha"] = (-np.log(1 - alpha)) ** (1 / 3)
    
    else:
        print(f"Unknown model: {model}")

# Define the model to use
model = 'A2'

# Dictionary to store dataframes
data_frames = {}

# Process each file and apply the FAlpha function
for file_path in file_paths:
    condition = process_file(file_path)
    if condition:
        FAlpha(data_frames, condition, model)
        output_file_path = f'C:/Users/kiran/OneDrive/Desktop/Internship/data/AlphaData/A2/HP{condition}_{model}.txt'
        data_frames[condition].to_csv(output_file_path, sep=';', index=False, encoding='latin1')
        print(f"Processed {condition}")

print("All files processed.")
