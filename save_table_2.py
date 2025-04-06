import os
import re
import pandas as pd
import glob

# Define dataset names in order
# datasets = [
#     "Flare-F", "Yeast5", "CarvGood", "CarGood", "Yeast5-ERL", "Glass", "ILPD", 
#     "Seed", "Wine", "Breast Cancer Wisconsin", "Diabetes", "Sonar", 
#     "Epileptic Seizure Recognition", "Student_dropout", "default of credit card clients"
# ]
# datasets = [
#     "Flare-F", "Yeast5", "CarvGood", "CarGood", "Glass", "ILPD", 
#     "Seed", "Wine", "Breast Cancer Wisconsin", "Diabetes", 
#     "Epileptic Seizure Recognition", "Student_dropout", "default of credit card clients"
# ]
datasets = [
    "Breast Cancer Wisconsin", "Diabetes", "Sonar", 
    "Epileptic Seizure Recognition", "Student_dropout", "default of credit card clients"
]

# Initialize dictionary to store results
results = {dataset: {"Adasyn": None} for dataset in datasets}

methodslike=["Adasyn"]

# Define pattern to extract times
pattern = r"======\[Dataset: (.*?) - (.*?) Method: (.*?)\]======.*?\[Time: ([\d.]+)\]"
# ======[Dataset: Sonar - Smote ENN Method: Sonar]======
# ======[Time: 1895.8165645599365]======

# Read each matching file
directory = "results_table"  # Update with the correct path
dataset_files = glob.glob(os.path.join(directory, "*_adasyn.txt_runtime_"))  # Find all dataset files

# for filename in os.listdir(directory):
#     if filename.endswith("_adaboost.txt_runtime_"):
#         dataset_name = filename.split("_adaboost.txt_runtime_")[0]  # Extract dataset name
#         print(dataset_name)
#         file_path = os.path.join(directory, filename)
#         with open(file_path, "r", encoding="utf-8") as file:
#             content = file.read()
#             times = re.findall(pattern, content)

#             # Assuming the first occurrence is Adaboost and the second is SmoteBoost
#             if dataset_name in results:
#                 if len(times) > 0:
#                     results[dataset_name]["Adaboost"] = times[0]
#                 if len(times) > 1:
#                     results[dataset_name]["SmoteBoost"] = times[1]
for dataset_file in dataset_files:
    with open(dataset_file, "r") as f:
        data = f.read()

    matches = re.findall(pattern, data, re.DOTALL)
    # print(matches)

    for dataset, method, dataset, time in matches:
        if method not in methodslike:
            continue
        print(dataset+" "+method)
        if dataset not in results:
            results[dataset] = {}
        results[dataset][method] = float(time)/1000
print(results)
# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(results, orient="index").reset_index()
df.columns = ["Dataset", "Rusboost"]

# Save to Excel
df.to_excel("adasyn_new_runtime_results.xlsx", index=False)

print("Excel file created: runtime_results.xlsx")
