import os
import numpy as np
import pandas as pd
from GA import genetic_algorithm

def run_multiple_times(file_path, evaluations_budget, runs=100, population_size=10):
    """
    Run the GA multiple times for a given dataset (file_path).
    For each run, record the best fitness found.
    Returns a list of best-fitness values for subsequent summary stats.
    """
    best_fitnesses = []

    system_name = os.path.basename(file_path)
    print(f"\nRunning GA for system: {system_name}")

    for i in range(runs):
        # We do not create a per-run CSV inside GA. Instead, we'll collect
        # each run's best fitness in this loop.
        best_solution, best_fitness = genetic_algorithm(
            file_path,
            evaluations_budget,
            output_file=None,  # no direct .csv from GA on each run
            population_size=population_size
        )
        best_fitnesses.append(best_fitness)
        print(f"Run {i + 1}: Best fitness = {best_fitness}")

    return best_fitnesses


def main():
    # Folder for the dataset CSVs
    datasets_folder = "datasets"
    # Folder for storing raw run data and summary
    output_folder = "GA_RawRunData"
    os.makedirs(output_folder, exist_ok=True)

    evaluations_budget = 100  # GA uses 100 evaluations (matching Random Search)
    runs = 100                # Number of times to run GA per dataset

    # This list of dictionaries will let us create a final summary CSV at the end
    results_summary = []

    # Process each CSV file in the datasets folder
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            
            # Collect best fitness from each run
            best_fitnesses = run_multiple_times(
                file_path,
                evaluations_budget=evaluations_budget,
                runs=runs,
                population_size=10
            )

            # Write raw data (each run's best fitness) to its own CSV
            # system_name_no_ext will match the naming in the random search code
            system_name_no_ext = os.path.splitext(file_name)[0]
            raw_data_file = os.path.join(
                output_folder, f"raw_data_{system_name_no_ext}_GA.csv"
            )

            df_raw = pd.DataFrame({
                "Run": range(1, runs + 1),
                "BestFitness": best_fitnesses
            })
            df_raw.to_csv(raw_data_file, index=False)
            print(f"Raw data saved to: {raw_data_file}")

            # Compute summary statistics
            avg_best = np.mean(best_fitnesses)
            std_best = np.std(best_fitnesses)

            # Append to the list for the final summary
            results_summary.append({
                "System": file_name,
                "Average Best Fitness": avg_best,
                "Standard Deviation": std_best
            })

            print("-" * 50)

    # Print and save final summary for all systems
    print("\nFinal Summary for All Systems:")
    for summary in results_summary:
        system = summary["System"]
        avg = summary["Average Best Fitness"]
        std = summary["Standard Deviation"]
        print(f"System: {system}")
        print(f"  Average Best Fitness: {avg}")
        print(f"  Standard Deviation:   {std}")
        print("-" * 30)

    # Write all summary stats to a single CSV file
    summary_df = pd.DataFrame(results_summary)
    summary_file_path = os.path.join(output_folder, "GA_summary_statistics.csv")
    summary_df.to_csv(summary_file_path, index=False)
    print(f"\nSummary statistics saved to: {summary_file_path}")


if __name__ == "__main__":
    main()
