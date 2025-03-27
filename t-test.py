import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind_from_stats

# Define file paths in their respective folders
ga_file = os.path.join("GA_RawRunData", "GA_summary_statistics.csv")
rs_file = os.path.join("RS_RawRunData", "RS_summary_statistics.csv")

# Load the summary statistics CSV files
ga_stats = pd.read_csv(ga_file)
rs_stats = pd.read_csv(rs_file)

# Assume each summary file is based on 100 runs
n_ga = 100
n_rs = 100

# Prepare a list to store results for each system
results = []

# Iterate over each system in the GA summary file (assuming the "System" column exists)
for idx, row in ga_stats.iterrows():
    system = row["System"]
    ga_mean = row["Average Best Fitness"]
    ga_sd = row["Standard Deviation"]

    # Get the corresponding row in the RS summary file
    rs_row = rs_stats[rs_stats["System"] == system]
    if rs_row.empty:
        print(f"System {system} not found in RS statistics. Skipping.")
        continue
    rs_mean = rs_row["Average Best Fitness"].values[0]
    rs_sd = rs_row["Standard Deviation"].values[0]

    # Conduct Welch's t-test
    t_stat, p_value = ttest_ind_from_stats(
        mean1=rs_mean, std1=rs_sd, nobs1=n_rs,
        mean2=ga_mean, std2=ga_sd, nobs2=n_ga,
        equal_var=False
    )

    # Determine significance (using alpha = 0.05)
    significance = "Yes" if p_value < 0.05 else "No"

    # Append the results as a dictionary
    results.append({
        "System": system,
        "RS_Mean": rs_mean,
        "RS_SD": rs_sd,
        "GA_Mean": ga_mean,
        "GA_SD": ga_sd,
        "p-value": p_value,
        "Significant (p < 0.05)": significance
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Print the results table
print(results_df)

# Save the results table to a CSV file
output_csv = "Welch_ttest_results.csv"
results_df.to_csv(output_csv, index=False)
print(f"Welch's t-test results saved to {output_csv}")
