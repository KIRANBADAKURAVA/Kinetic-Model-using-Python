import pandas as pd

# Load the data
file_path = r'C:\Users\kiran\OneDrive\Desktop\Model Repo\Kinetic-Model-using-Python\data\AlphaData\HP4_30_D1.txt'  # Replace with your actual file path
data = pd.read_csv(file_path, delimiter=';')

# Function to find the row with the alpha value closest to each target value
def find_closest_rows(data, target_alphas):
    closest_rows = pd.DataFrame()
    for alpha in target_alphas:
        closest_row = data.iloc[(data['Alpha'] - alpha).abs().argmin()]
        closest_rows = pd.concat([closest_rows, closest_row.to_frame().T], ignore_index=True)
    return closest_rows

# Extend the range of target alphas to cover from 0.025 to 1 with a step of 0.005
target_alphas = [i/1000 for i in range(25, 1001, 5)]

# Find the closest rows for the extended range of target alphas
closest_rows_extended = find_closest_rows(data, target_alphas)

# Remove duplicate rows from the DataFrame
closest_rows_unique = closest_rows_extended.drop_duplicates()

# Round off alpha values to 2 decimal places
closest_rows_unique['Alpha'] = closest_rows_unique['Alpha'].round(2)

# Remove duplicates based on the rounded alpha values
closest_rows_final = closest_rows_unique.drop_duplicates(subset=['Alpha'])

# Display the final data
print(closest_rows_final)

# Optionally, save the final dataset to a CSV file
closest_rows_final.to_csv('filtered_alpha_values.csv', index=False)
