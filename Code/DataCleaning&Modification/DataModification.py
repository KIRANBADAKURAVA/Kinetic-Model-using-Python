
import pandas as pd
import glob

input_file_path = r"C:\Users\kiran\OneDrive\Desktop\Internship\data\HP1_30.txt"
output_file_path = r"C:\Users\kiran\OneDrive\Desktop\Internship\data\HP1_30.txt"


with open(input_file_path, 'r') as file:
    lines = file.readlines()


start_index = 0
for i, line in enumerate(lines):
    if line.startswith('118.78200;7.75000;99.94843;1'):
        start_index = i
        break

# Extract and modify the relevant data
modified_lines = []
start_time = 0.0
start_mass = 100.0

for line in lines[start_index:]:
    parts = line.split(';')
    if len(parts) != 4:
        continue

    temperature = float(parts[0])
    time = float(parts[1]) - float(lines[start_index].split(';')[1])
    mass = float(parts[2]) / float(lines[start_index].split(';')[2]) * start_mass
    segment = parts[3].strip()

    modified_lines.append(f"{temperature:.5f};{time:.5f};{mass:.5f};{segment}\n")

# Write the modified data to the output file
with open(output_file_path, 'w') as file:
    file.writelines(modified_lines)

print(f"Modified data has been written to {output_file_path}")
