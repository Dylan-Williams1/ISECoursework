import os
import numpy as np
import pandas as pd

def random_search_single_run(file_path, budget):
    """
    Conduct a single run of Random Search and return (best_solution, best_performance).
    """
    data = pd.read_csv(file_path)

    # Identify the columns for configurations and performance
    config_columns = data.columns[:-1]
    performance_column = data.columns[-1]

    # Determine if this is a maximization or minimization problem (by dataset name)
    system_name = os.path.basename(file_path).split('.')[0]
    # Example rule: If system_name == 'mySystem' -> set maximization = True
    # In your example code, you used '---' as a placeholder. Adjust as you see fit:
    if system_name.lower() == "---":
        maximization = True
    else:
        maximization = False

    # Calculate a 'worst_value' for configurations not in the dataset
    if maximization:
        worst_value = data[performance_column].min() / 2
    else:
        worst_value = data[performance_column].max() * 2

    # Initialize best performance trackers
    best_performance = -np.inf if maximization else np.inf
    best_solution = []

    # For each evaluation under the budget
    for _ in range(budget):
        # Randomly sample one configuration
        sampled_config = [int(np.random.choice(data[col].unique()))
                          for col in config_columns]

        # Check if that configuration exists in dataset
        matched_row = data.loc[(data[config_columns]
                               == pd.Series(sampled_config, index=config_columns)).all(axis=1)]

        if not matched_row.empty:
            performance = matched_row[performance_column].iloc[0]
        else:
            # Invalid / missing config
            performance = worst_value

        # Update the best found
        if maximization:
            if performance > best_performance:
                best_performance = performance
                best_solution = sampled_config
        else:
            if performance < best_performance:
                best_performance = performance
                best_solution = sampled_config

    return best_solution, best_performance

def run_multiple_times_random_search(file_path, budget=100, runs=100):
    """
    Run the random search multiple times for a given dataset, each time returning
    the best performance. Return a list of best_performance values (length 'runs').
    """
    best_fitnesses = []
    system_name = os.path.basename(file_path)
    print(f"\nRunning Random Search for system: {system_name}")

    for i in range(runs):
        best_solution, best_performance = random_search_single_run(file_path, budget)
        best_fitnesses.append(best_performance)
        print(f"Run {i+1}: Best performance = {best_performance}")

    return best_fitnesses

def main():
    datasets_folder = "datasets"
    output_folder = "RS_RawRunData"  # Folder for storing raw run data & summary stats
    os.makedirs(output_folder, exist_ok=True)

    runs = 100
    budget = 100

    results_summary = []

    # Process each CSV file
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)

            # 1) Collect best fitness from each run
            best_fitnesses = run_multiple_times_random_search(
                file_path, 
                budget=budget, 
                runs=runs
            )

            # 2) Write raw data for each run to a CSV
            system_name_no_ext = os.path.splitext(file_name)[0]
            raw_data_file = os.path.join(
                output_folder,
                f"raw_data_{system_name_no_ext}_RS.csv"
            )
            df_raw = pd.DataFrame({
                "Run": range(1, runs + 1),
                "BestFitness": best_fitnesses
            })
            df_raw.to_csv(raw_data_file, index=False)
            print(f"Raw run data saved to: {raw_data_file}")

            # 3) Compute summary statistics
            avg_best = np.mean(best_fitnesses)
            std_best = np.std(best_fitnesses)
            results_summary.append({
                "System": file_name,
                "Average Best Fitness": avg_best,
                "Standard Deviation": std_best
            })
            print("-" * 50)

    # Print final summary
    print("\nFinal Summary for All Systems (Random Search):")
    for summary in results_summary:
        system = summary["System"]
        avg = summary["Average Best Fitness"]
        std = summary["Standard Deviation"]
        print(f"System: {system}")
        print(f"  Average Best Fitness: {avg}")
        print(f"  Standard Deviation:   {std}")
        print("-" * 30)

    # Write summary stats to CSV
    summary_df = pd.DataFrame(results_summary)
    summary_file_path = os.path.join(output_folder, "RS_summary_statistics.csv")
    summary_df.to_csv(summary_file_path, index=False)
    print(f"\nSummary statistics saved to {summary_file_path}")

if __name__ == "__main__":
    main()
