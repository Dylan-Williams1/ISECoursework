import os
import numpy as np
import pandas as pd
from GA import genetic_algorithm

def run_multiple_times(file_path, evaluations_budget, runs, population_size=10):
    best_fitnesses = []

    print(f"\nRunning GA for system: {os.path.basename(file_path)}")

    for i in range(runs):
        best_solution, best_fitness = genetic_algorithm(
            file_path, evaluations_budget, None,
            population_size=population_size
        )
        best_fitnesses.append(best_fitness)
        print(f"Run {i+1}: | Best fitness: {best_fitness}")

    avg_best = np.mean(best_fitnesses)
    std_best = np.std(best_fitnesses)
    return avg_best, std_best


def main():
    datasets_folder = "datasets"
    evaluations_budget = 100  # Total evaluations per GA run
    runs = 100  # Number of times to run GA per system
    
    results_summary = {}  # To store summary results for all systems
    
    # Process each CSV file in the datasets folder.
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            avg, std = run_multiple_times(file_path, evaluations_budget, runs=runs, 
                                          population_size=10)
            results_summary[file_name] = {"Average Best Fitness": avg, "Standard Deviation": std}
            print("-" * 50)
    
    # Final summary for all systems.
    print("\nFinal Summary for All Systems:")
    for system, stats in results_summary.items():
        print(f"System: {system}")
        print(f"  Average Best Fitness: {stats['Average Best Fitness']}")
        print(f"  Standard Deviation: {stats['Standard Deviation']}")
        print("-" * 30)

if __name__ == "__main__":
    main()
