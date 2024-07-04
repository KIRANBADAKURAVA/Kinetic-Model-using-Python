import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
file_pattern = 'C:/Users/kiran/OneDrive/Desktop/Internship/data/AlphaData/HP*.txt'
file_paths = glob.glob(file_pattern)

# Dictionary to store dataframes with condition as the key
data_frames = {}

# Read each file and extract conditions
for file_path in file_paths:
    filename = file_path.split('\\')[-1].split('.')[0]
    conditions = filename.split('_')
    condition_1 = conditions[0][2:]  # Extracting the first condition
    condition_2 = conditions[1]      # Extracting the second condition
    condition = f'H{condition_1}_{condition_2}'
    
    try:
        # Read the file into a dataframe
        df = pd.read_csv(file_path, sep=';', header=0, encoding='latin1')
        data_frames[condition] = df
    except pd.errors.ParserError as e:
        print(f"ParserError in file {file_path}: {e}")


for key in data_frames :
    for c in data_frames[key].columns:
        print(c)

# x1=[273.47501,327.59201,310.22198]
# x2=[181.20599,225.54601,221.457]
# x3=[354.85101,402.159,403.48001]
# y=[10,20,30]

# plt.plot(x1,y)
# plt.plot(x2,y)
# plt.plot(x3,y)
# plt.show()