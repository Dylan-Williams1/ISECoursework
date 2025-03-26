import pandas as pd
import numpy as np
import os

def load_data(file_path):
    data = pd.read_csv(file_path)
    config_columns = data.columns[:-1]
    performance_column = data.columns[-1]
    
    system_name = os.path.basename(file_path).split('.')[0]
    maximization = True if system_name.lower() == "---" else False

    if maximization:
        worst_value = data[performance_column].min() / 2
    else:
        worst_value = data[performance_column].max() * 2

    return data, config_columns, performance_column, maximization, worst_value

def evaluate(individual, data, config_columns, performance_column, worst_value):
    mask = (data[config_columns] == pd.Series(individual, index=config_columns)).all(axis=1)
    matched_row = data.loc[mask]
    return matched_row[performance_column].iloc[0] if not matched_row.empty else worst_value

def initialize_population(population_size, data, config_columns):
    return [[int(np.random.choice(data[col].unique())) for col in config_columns]
            for _ in range(population_size)]

def tournament_selection(population, fitnesses, maximization, k=2):
    indices = np.random.choice(len(population), k, replace=False)
    best_index = indices[0]
    for idx in indices[1:]:
        if (maximization and fitnesses[idx] > fitnesses[best_index]) or \
           (not maximization and fitnesses[idx] < fitnesses[best_index]):
            best_index = idx
    return population[best_index]

def crossover(parent1, parent2, num_points=2):
    n = len(parent1)
    if n < 2:
        return parent1.copy(), parent2.copy()

    effective_points = min(num_points, n - 1)
    points = sorted(np.random.choice(range(1, n), effective_points, replace=False))

    child1, child2 = parent1.copy(), parent2.copy()
    start, swap = 0, False
    for point in points:
        if swap:
            child1[start:point], child2[start:point] = child2[start:point], child1[start:point]
        swap = not swap
        start = point

    if swap:
        child1[start:], child2[start:] = child2[start:], child1[start:]
    return child1, child2

def mutate(individual, data, config_columns, mutation_rate):
    return [int(np.random.choice(data[col].unique())) if np.random.rand() < mutation_rate else gene
            for gene, col in zip(individual, config_columns)]

def genetic_algorithm(file_path, evaluations_budget, output_file, population_size=10):
    generations = evaluations_budget // population_size
    data, config_columns, perf_col, maximization, worst_value = load_data(file_path)

    population = initialize_population(population_size, data, config_columns)
    fitnesses = [evaluate(ind, data, config_columns, perf_col, worst_value) for ind in population]

    best_individual = population[np.argmax(fitnesses) if maximization else np.argmin(fitnesses)]
    best_fitness = max(fitnesses) if maximization else min(fitnesses)

    search_results = []

    for gen in range(generations):
        mutation_rate = max(0.1, 0.7 * np.exp(-gen / generations))
        new_population = []

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses, maximization)
            parent2 = tournament_selection(population, fitnesses, maximization)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, data, config_columns, mutation_rate)
            child2 = mutate(child2, data, config_columns, mutation_rate)
            new_population.extend([child1, child2][:population_size - len(new_population)])

        population = new_population
        fitnesses = [evaluate(ind, data, config_columns, perf_col, worst_value) for ind in population]

        for ind, fit in zip(population, fitnesses):
            if (maximization and fit > best_fitness) or (not maximization and fit < best_fitness):
                best_individual, best_fitness = ind, fit
            search_results.append(ind + [fit])

    pd.DataFrame(search_results, columns=list(config_columns) + ["Performance"]).to_csv(output_file, index=False)
    return best_individual, best_fitness

def main():
    datasets_folder = "datasets"
    output_folder = "GAsearch_results"
    os.makedirs(output_folder, exist_ok=True)

    evaluations_budget = 100
    results = {}

    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_GA_search_results.csv")
            best_solution, best_performance = genetic_algorithm(file_path, evaluations_budget, output_file)
            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance
            }

    for system, result in results.items():
        print(f"System: {system}")
        print(f"  Best Solution: [{', '.join(map(str, result['Best Solution']))}]")
        print(f"  Best Performance: {result['Best Performance']}")

if __name__ == "__main__":
    main()
