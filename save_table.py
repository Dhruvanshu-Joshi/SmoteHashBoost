# import pandas as pd
# import re
# import glob
# import os

# # File paths
# results_dir = "results_HUE/"
# results_dir1 = "results/"
# excel_file = os.path.join(results_dir1, "datatable_hashboost.xlsx")
# dataset_files = glob.glob(os.path.join(results_dir, "*_HUE.txt"))  # Find all dataset files

# # Load the Excel file using openpyxl
# xls = pd.ExcelFile(excel_file, engine="openpyxl")
# df = pd.read_excel(xls, sheet_name="Sheet1")

# # Extract dataset names (ignoring the first two header rows)
# dataset_names = df.iloc[2:, 1].dropna().astype(str).tolist()

# # Extract method names and their corresponding column indices
# method_columns = {}
# for col_index in range(2, len(df.columns), 5):  # Step by 5 for each method block
#     method_name = str(df.iloc[0, col_index]).strip()  # Extract method name from row 0
#     if method_name and method_name != "NaN":
#         method_columns[method_name] = col_index

# # Regex pattern to extract method names and best metric values
# pattern = r"======\[Dataset: (.*?) - Method: (.*?)\]======.*?\[Best\] Accuracy: (\d+\.\d+), AUC: (\d+\.\d+), F1: (\d+\.\d+), AP: (\d+\.\d+), Gmean: (\d+\.\d+)"
# # ======[Dataset: Breast Cancer Wisconsin - SmoteHashBoost Method: Random]======
# # [Avg] Accuracy: 0.9404, AUC: 0.9380, F1: 0.9207, AP: 0.8760, Gmean: 0.9376
# # [Std] Accuracy: 0.0053, AUC: 0.0057, F1: 0.0069, AP: 0.0111, Gmean: 0.0058
# # [Best] Accuracy: 0.9526, AUC: 0.9497, F1: 0.9361, AP: 0.9004, Gmean: 0.9495

# # Create a dictionary to store extracted values
# results = {}
# for dataset_file in dataset_files:
#     with open(dataset_file, "r") as f:
#         data = f.read()

#     matches = re.findall(pattern, data, re.DOTALL)

#     for dataset, method, acc, auc, f1, ap, gmean in matches:
#         if dataset not in results:
#             results[dataset] = {}
#         results[dataset][method] = {
#             "Accuracy": float(acc),
#             "AUC": float(auc),
#             "F1": float(f1),
#             "AP": float(ap),
#             "Gmean": float(gmean),
#         }
# print(matches)

# ap=[[]]
# gm=[[]]
# f1=[[]]
# name=[]
# i=0

# # for index, row in df.iterrows():
# #     dataset_name = str(row["Unnamed: 1"]).strip()
# #     print(dataset_name)
# #     if dataset_name in results:
# #         # for method, col_index in method_columns.items():
# #         #     if method in results[dataset_name]:
# #                 ap.append()


# # Fill dataset values into the DataFrame
# for index, row in df.iterrows():
#     dataset_name = str(row["Unnamed: 1"]).strip()
#     print(dataset_name)
#     if dataset_name in results:
#         name.append(dataset_name)
#         for method, col_index in method_columns.items():
#             if method in results[dataset_name]:
#                 # Fill values
#                 df.iloc[index, col_index] = results[dataset_name][method]["Accuracy"]
#                 df.iloc[index, col_index + 1] = results[dataset_name][method]["AUC"]
#                 df.iloc[index, col_index + 2] = results[dataset_name][method]["F1"]
#                 f1[i].append(results[dataset_name][method]["F1"])
#                 df.iloc[index, col_index + 3] = results[dataset_name][method]["AP"]
#                 ap[i].append(results[dataset_name][method]["AP"])
#                 df.iloc[index, col_index + 4] = results[dataset_name][method]["Gmean"]
#                 gm[i].append(results[dataset_name][method]["Gmean"])
        # ap.append([])
        # f1.append([])
        # gm.append([])
        # i=i+1
# # Save the updated Excel file
# updated_file_path = os.path.join(results_dir, "new_Updated_" + os.path.basename(excel_file))
# df.to_excel(updated_file_path, index=False, engine="openpyxl")

# print(f"Updated file saved at: {updated_file_path}")

# output_file_f1 = "results_plot/F1_result_HUE.txt"
# output_file_gm = "results_plot/GM_result_HUE.txt"
# output_file_ap = "results_plot/AP_result_HUE.txt"
# for x in range(0,i):
#     print(f"{max(f1[x])}")
#     # print(f"ap : {max(ap[x])}")
#     # print(f"gm : {max(gm[x])}")
#     with open(output_file_f1, 'a') as file:
#         file.write("\n")
#         file.write(f"======[Dataset: {name[x]}]======\n")
#         # file.write("\n")
#         file.write(f"{max(f1[x])}")
#         file.write("\n")
# print()

# for x in range(0,i):
#     # print(f"f1 : {max(f1[x])}")
#     with open(output_file_ap, 'a') as file:
#         file.write("\n")
#         file.write(f"======[Dataset: {name[x]}]======\n")
#         # file.write("\n")
#         file.write(f"{max(ap[x])}")
#         file.write("\n")
#     # print(f"gm : {max(gm[x])}")
# print()

# for x in range(0,i):
#     # print(f"f1 : {max(f1[x])}")
#     # print(f"ap : {max(ap[x])}")
#     with open(output_file_gm, 'a') as file:
#         file.write("\n")
#         file.write(f"======[Dataset: {name[x]}]======\n")
#         # file.write("\n")
#         file.write(f"{max(gm[x])}")
#         file.write("\n")
#     print(f"{max(gm[x])}")
# print()
# # with open(output_file, 'a') as file:
# #         file.write("\n")
# #         file.write(f"======[Dataset: {name[x]}]======\n")
# # with open(output_file, 'a') as file:
# #     file.write(OUTPUT.format("Avg", *avg_metrics) + "\n")
# #     file.write(OUTPUT.format("Std", *std_metrics) + "\n")
# #     file.write(OUTPUT.format("Best", *best_metrics) + "\n")


# import pandas as pd
# import re
# import os

# # File paths
# results_dir = "results/"
# excel_file = os.path.join(results_dir, "datatable_smoteboost.xlsx")

# # Load the Excel file using openpyxl
# xls = pd.ExcelFile(excel_file, engine="openpyxl")
# df = pd.read_excel(xls, sheet_name="Sheet1")

# # Extract dataset names (ignoring the first two header rows)
# dataset_names = df.iloc[2:, 1].dropna().astype(str).tolist()

# # Regex pattern to extract method names and best metric values
# pattern = r"======\[Dataset: (.*?) - Method: (.*?)\]======.*?\[Std\] Accuracy: (\d+\.\d+), AUC: (\d+\.\d+), F1: (\d+\.\d+), AP: (\d+\.\d+), Gmean: (\d+\.\d+)"

# # Process all files ending with 'smoteboost.txt'
# for file_name in os.listdir(results_dir):
#     if file_name.endswith("smotebooost.txt"):
#         dataset_file = os.path.join(results_dir, file_name)

#         # Create a dictionary to store extracted values
#         results = {}
#         with open(dataset_file, "r") as f:
#             data = f.read()

#         matches = re.findall(pattern, data, re.DOTALL)

#         for dataset, method, acc, auc, f1, ap, gmean in matches:
#             if dataset not in results:
#                 results[dataset] = {}
#             results[dataset][method] = {
#                 "Accuracy": float(acc),
#                 "AUC": float(auc),
#                 "F1": float(f1),
#                 "AP": float(ap),
#                 "Gmean": float(gmean),
#             }

#         # Fill dataset values into the DataFrame
#         for index, row in df.iterrows():
#             # print('aaa')
#             dataset_name = str(row["Unnamed: 1"]).strip()
#             # print(dataset_name)
#             if dataset_name in results:
#                 print("iq")
#                 print(results[dataset_name])
#                 for method in results[dataset_name]:
#                     print(method)
#                     print("gq")
#                     # method_cols = df.columns[df.iloc[0] == method].tolist()
#                     # if method_cols:
#                     #     col_index = df.columns.get_loc(method_cols[0])
#                     # Fill values
#                     col_index=2
#                     df.iloc[index, col_index] = results[dataset_name][method]["Accuracy"]
#                     df.iloc[index, col_index + 1] = results[dataset_name][method]["AUC"]
#                     df.iloc[index, col_index + 2] = results[dataset_name][method]["F1"]
#                     df.iloc[index, col_index + 3] = results[dataset_name][method]["AP"]
#                     df.iloc[index, col_index + 4] = results[dataset_name][method]["Gmean"]

#         # print(results)

# # Save the updated Excel file
# updated_file_path = os.path.join(results_dir, "Updated_" + os.path.basename(excel_file))
# df.to_excel(updated_file_path, index=False, engine="openpyxl")

# print(f"Updated file saved at: {updated_file_path}")


# import pandas as pd
# import re
# import glob
# import os

# # File paths
# results_dir = "results_adasyn/"
# results_dir1 = "results/"
# excel_file = os.path.join(results_dir1, "datatable_smoteboost.xlsx")
# dataset_files = glob.glob(os.path.join(results_dir, "*_adasyn.txt"))  # Find all dataset files

# # Load the Excel file using openpyxl
# xls = pd.ExcelFile(excel_file, engine="openpyxl")
# df = pd.read_excel(xls, sheet_name="Sheet1")

# # print(df.to_string())
# # assert 0

# # Extract dataset names (ignoring the first two header rows)
# dataset_names = df.iloc[2:, 1].dropna().astype(str).tolist()
# # print(dataset_names)

# # Extract method names and their corresponding column indices
# method_columns = {}
# # print(len(df.columns))
# for col_index in range(2, len(df.columns), 5):  # Step by 5 for each method block
#     method_name = str(df.iloc[0, col_index]).strip()  # Extract method name from row 0
#     # print(method_name)
#     if method_name and method_name != "NaN":
#         method_columns[method_name] = col_index

# # print(method_columns)

# # Regex pattern to extract method names and best metric values
# pattern = r"======\[Dataset: (.*?) - Adasyn Method: (.*?)\]======.*?\[Best\] Accuracy: (\d+\.\d+), AUC: (\d+\.\d+), F1: (\d+\.\d+), AP: (\d+\.\d+), Gmean: (\d+\.\d+)"
# # ======[Dataset: Yeast5-ERL - SmoteTomek Method: Yeast5-ERL]======
# # ======[Dataset: Breast Cancer Wisconsin - Method: Reciprocal]======
# # ======[Dataset: CarGood - Method: CarGood]======
# # [Avg] Accuracy: 0.9875, AUC: 0.9215, F1: 0.8448, AP: 0.7320, Gmean: 0.9166
# # [Std] Accuracy: 0.0029, AUC: 0.0232, F1: 0.0350, AP: 0.0535, Gmean: 0.0256
# # [Best] Accuracy: 0.9913, AUC: 0.9391, F1: 0.8900, AP: 0.8010, Gmean: 0.9367
# # ======[Dataset: Breast Cancer Wisconsin - Adaboost Method: Breast Cancer Wisconsin]======
# # [Avg] Accuracy: 0.9251, AUC: 0.9222, F1: 0.9007, AP: 0.8464, Gmean: 0.9217
# # [Std] Accuracy: 0.0081, AUC: 0.0081, F1: 0.0103, AP: 0.0157, Gmean: 0.0082
# # [Best] Accuracy: 0.9367, AUC: 0.9371, F1: 0.9175, AP: 0.8662, Gmean: 0.9368

# # Create a dictionary to store extracted values
# results = {}
# # print(dataset_files)
# for dataset_file in dataset_files:
#     with open(dataset_file, "r") as f:
#         data = f.read()

#     matches = re.findall(pattern, data, re.DOTALL)
#     # print(matches)

#     for dataset, method, acc, auc, f1, ap, gmean in matches:
#         print(dataset+" "+method)
#         if dataset not in results:
#             results[dataset] = {}
#         results[dataset][method] = {
#             "Accuracy": float(acc),
#             "AUC": float(auc),
#             "F1": float(f1),
#             "AP": float(ap),
#             "Gmean": float(gmean),
#         }
#         # results[dataset] = {
#         #     "Accuracy": float(acc),
#         #     "AUC": float(auc),
#         #     "F1": float(f1),
#         #     "AP": float(ap),
#         #     "Gmean": float(gmean),
#         # }
# # assert 0)
# print(results)
# # print(matches)
# f1=[[]]
# ap=[[]]
# gm=[[]]
# i=0
# name=[]
# # Fill dataset values into the DataFrame
# for index, row in df.iterrows():
#     dataset_name = str(row["Unnamed: 1"]).strip()
#     name.append(dataset_name)
#     print(dataset_name)
#     if dataset_name in results:
#         # # print(dataset_name)
#         # for method, col_index in method_columns.items():
#         #     # print(method)
#         #     if method in results[dataset_name]:
#         #     # Fill values
#         col_index = 2
#         method=dataset_name
#         df.iloc[index, col_index] = results[dataset_name][method]["Accuracy"]
#         df.iloc[index, col_index + 1] = results[dataset_name][method]["AUC"]
#         df.iloc[index, col_index + 2] = results[dataset_name][method]["F1"]
#         f1[i].append(results[dataset_name][method]["F1"])
#         df.iloc[index, col_index + 3] = results[dataset_name][method]["AP"]
#         ap[i].append(results[dataset_name][method]["AP"])
#         df.iloc[index, col_index + 4] = results[dataset_name][method]["Gmean"]
#         gm[i].append(results[dataset_name][method]["Gmean"])
#     ap.append([])
#     f1.append([])
#     gm.append([])
#     i=i+1
# print(name)
# print(f1)
# # f1=f1[1:13]
# # ap=ap[1:13]
# # gm=gm[1:13]
# # name=name[1:13]
# print(name)
# print("i")

# # Save the updated Excel file
# updated_file_path = os.path.join(results_dir, "newUpdated_smoteboost_" + os.path.basename(excel_file))
# df.to_excel(updated_file_path, index=False, engine="openpyxl")

# print(f"Updated file saved at: {updated_file_path}")


# output_file_f1 = "results_plot/F1_result_AS.txt"
# output_file_gm = "results_plot/GM_result_AS.txt"
# output_file_ap = "results_plot/AP_result_AS.txt"
# # for x in range(1,i-1):
# #     # print(f"{max(f1[x])}")
# #     # print(f"ap : {max(ap[x])}")
# #     # print(f"gm : {max(gm[x])}")
# #     with open(output_file_f1, 'a') as file:
# #         file.write("\n")
# #         file.write(f"======[Dataset: {name[x]}]======\n")
# #         # file.write("\n")
# #         if len(f1[x])==0:
# #             file.write("0")
# #         else:
# #             file.write(f"{max(f1[x])}")
# #         file.write("\n")
# # print()

# # for x in range(1,i-1):
# #     # print(f"f1 : {max(f1[x])}")
# #     with open(output_file_ap, 'a') as file:
# #         file.write("\n")
# #         file.write(f"======[Dataset: {name[x]}]======\n")
# #         # file.write("\n")
# #         if len(ap[x])==0:
# #             file.write("0")
# #         else:
# #             file.write(f"{max(ap[x])}")
# #         file.write("\n")
# #     # print(f"gm : {max(gm[x])}")
# # print()

# for x in range(1,i-1):
#     # print(x)
#     # print(f"f1 : {max(f1[x])}")
#     # print(f"ap : {max(ap[x])}")
#     with open(output_file_gm, 'a') as file:
#         file.write("\n")
#         file.write(f"======[Dataset: {name[x]}]======\n")
#         # file.write("\n")
#         if len(f1[x])==0:
#             file.write("0")
#         else:
#             file.write(f"{max(gm[x])}")
#         file.write("\n")
#     # print(f"{max(gm[x])}")
# print()


import pandas as pd
import re
import glob
import os

# File paths
results_dir = "results/"
excel_file = os.path.join(results_dir, "hashboost_table.xlsx")
dataset_files = glob.glob(os.path.join(results_dir, "*TYME*"))  # Find all dataset files

# Load the Excel file using openpyxl
xls = pd.ExcelFile(excel_file, engine="openpyxl")
df = pd.read_excel(xls, sheet_name="Sheet1")

# Extract dataset names (ignoring the first two header rows)
dataset_names = df.iloc[2:, 1].dropna().astype(str).tolist()

# Extract method names and their corresponding column indices
method_columns = {}
for col_index in range(2, len(df.columns), 1):  # Step by 5 for each method block
    method_name = str(df.iloc[0, col_index]).strip()  # Extract method name from row 0
    if method_name and method_name != "NaN":
        method_columns[method_name] = col_index

# Regex pattern to extract method names and best metric values
pattern = r"======\[Dataset: (.*?) - Method: (.*?)\]======.*?\[Time: ([\d.]+)\]"
# r"======\[Dataset: (.*?) - (.*?) Method: (.*?)\]s

# Create a dictionary to store extracted values
results = {}
for dataset_file in dataset_files:
    with open(dataset_file, "r") as f:
        data = f.read()

    matches = re.findall(pattern, data, re.DOTALL)

    for dataset, method, tme in matches:
        if dataset not in results:
            results[dataset] = {}
        results[dataset][method] = {
            "Time": float(tme)/1000,
            # "AUC": float(auc),
            # "F1": float(f1),
            # "AP": float(ap),
            # "Gmean": float(gmean),
        }
# print(matches)
# print(results)
# Fill dataset values into the DataFrame
for index, row in df.iterrows():
    # print(row)
    dataset_name = str(row["Dataset"]).strip()
    print(dataset_name)
    if dataset_name in results:
        for method, col_index in method_columns.items():
            # print(results[dataset_name])
            # print(method)
            if method in results[dataset_name]:
                # print(method)
                # Fill values
                df.iloc[index, col_index] = results[dataset_name][method]["Time"]
                # df.iloc[index, col_index + 1] = results[dataset_name][method]["AUC"]
                # df.iloc[index, col_index + 2] = results[dataset_name][method]["F1"]
                # df.iloc[index, col_index + 3] = results[dataset_name][method]["AP"]
                # df.iloc[index, col_index + 4] = results[dataset_name][method]["Gmean"]

# Save the updated Excel file
updated_file_path = os.path.join(results_dir, "Updated_hashboost_" + os.path.basename(excel_file))
df.to_excel(updated_file_path, index=False, engine="openpyxl")

print(f"Updated file saved at: {updated_file_path}")
