import os
import numpy as np
from main import random_search  # Import your random search function

def run_multiple_times(file_path, budget, runs):
    """
    Run the random search for a given system 'runs' times.
    For each run, print the best solution and its performance.
    Returns the average best performance and its standard deviation.
    """
    best_fitnesses = []
    
    system_name = os.path.basename(file_path)
    print(f"\nRunning Random Search for system: {system_name}")
    
    for i in range(runs):
        # We pass None as the output_file so that no CSV is written.
        best_solution, best_fitness = random_search(file_path, budget, None)
        best_fitnesses.append(best_fitness)
        print(f"Run {i+1} | Best performance: {best_fitness}")
    
    avg_best = np.mean(best_fitnesses)
    std_best = np.std(best_fitnesses)
    return avg_best, std_best

def main():
    datasets_folder = "datasets"
    budget = 100  # Number of evaluations per run
    runs = 100     # Number of runs per system
    
    results_summary = {}
    
    # Process each CSV file in the datasets folder.
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            avg, std = run_multiple_times(file_path, budget, runs=runs)
            results_summary[file_name] = {"Average Best Performance": avg, "Standard Deviation": std}
            print("-" * 70)
    
    # Final summary output for all systems.
    print("\nFinal Summary for All Systems:")
    for system, stats in results_summary.items():
        print(f"System: {system}")
        print(f"  Average Best Performance: {stats['Average Best Performance']}")
        print(f"  Standard Deviation: {stats['Standard Deviation']}")
        print("-" * 30)

if __name__ == "__main__":
    main()
