# import os
# import re

# # Directory containing the AP files
# directory = 'results_plot/'  # Change this if files are in a different directory

# # Dictionaries to store results
# results = {}
# datasets_dict = {}

# # Iterate through all files in the directory
# for filename in os.listdir(directory):
#     if filename.startswith('F1') and filename.endswith('.txt'):
#         key = filename[-8:-4]  # Extracting last 4 characters before '.txt'
        
#         with open(os.path.join(directory, filename), 'r') as file:
#             content = file.read()
            
#             # Extract all floating point numbers using regex
#             values = re.findall(r'\d+\.\d+', content)
            
#             # Extract all dataset names using regex
#             datasets = re.findall(r'\[Dataset: (.*?)\]', content)
            
#             # Convert the list of values to a comma-separated string
#             csv_values = ','.join(values)
            
#             # Add to the dictionaries
#             results[key] = csv_values
#             datasets_dict[key] = datasets

# # Print the final dictionaries
# print(results)
# print(datasets_dict)

import os
import re
import pandas as pd

# Directory containing the AP files
directory = 'results_plot/'  # Change this if files are in a different directory

# Dictionaries to store results
results = {}
datasets_dict = {}

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.startswith('GM') and filename.endswith('.txt'):
        key = filename[-8:-4]  # Extracting last 4 characters before '.txt'
        # print(key)
        
        with open(os.path.join(directory, filename), 'r') as file:
            content = file.read()
            
            # Extract all floating point numbers using regex
            values = re.findall(r'\d+\.\d+', content)
            
            # Extract all dataset names using regex
            datasets = re.findall(r'\[Dataset: (.*?)\]', content)
            
            # Convert the list of values to a comma-separated string
            csv_values = ','.join(values)
            
            # Add to the dictionaries
            results[key] = csv_values
            datasets_dict[key] = datasets

# print(datasets_dict)

# Create DataFrame
plotdata = pd.DataFrame(
    {key: list(map(float, value.split(','))) for key, value in results.items()},
    index=list(datasets_dict[list(datasets_dict.keys())[0]])  # Assuming all files have the same datasets
)

# Save DataFrame to CSV
plotdata.to_csv('output_gm.csv')

# Save CSV values as list in a text file
with open('gm_csv_list.txt', 'w') as f:
    for key, csv_values in results.items():
        f.write(f"{key}: {csv_values}\n")

# Print the DataFrame
print(plotdata)

