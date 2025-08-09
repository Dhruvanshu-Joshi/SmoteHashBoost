import os
import re
import pandas as pd

# Directory where all your result files are stored
results_dir = "results_table"

# Output dictionary to collect the data
data = []

# Regex to extract values from [Std] line
std_pattern = re.compile(r'\[Std\].*?F1:\s*([\d.]+),\s*AP:\s*([\d.]+),\s*Gmean:\s*([\d.]+)')

# Loop through all files in the directory
for filename in os.listdir(results_dir):
    if filename.startswith("extra_metrics_") and filename.endswith(".txt"):
        # Extract dataset name and method name from filename
        match = re.match(r"extra_metrics_(.*?)_result_avg_precision_(.*?)\.txt", filename)
        if match:
            dataset = match.group(1)
            method = match.group(2)

            # Read the file and extract [Std] values
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                content = f.read()
                std_match = std_pattern.search(content)
                if std_match:
                    f1, ap, gmean = std_match.groups()
                    data.append({
                        "Dataset": dataset,
                        "Method": method,
                        "Std_F1": float(f1),
                        "Std_AP": float(ap),
                        "Std_Gmean": float(gmean)
                    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_file = "std_metrics_summary.csv"
df.to_csv(output_file, index=False)

print(f"Extracted and saved results to {output_file}")
