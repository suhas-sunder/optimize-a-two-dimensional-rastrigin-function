import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from utils.shared_rastrigin_fcn import rastrigin  # Rastrigin function implementation

# Set random seed for reproducibility
random.seed(42)

# Define fitness strategy (minimize) and individual structure
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Single-objective minimization
creator.create("Individual", list, fitness=creator.FitnessMin)  # Each individual is a list with fitness

# Initialize genetic operators
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5.12, 5.12)  # Random float in search space
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 2)  # 2D individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # List of individuals

# Register evaluation and genetic operators
toolbox.register("evaluate", lambda ind: (rastrigin(np.array(ind)),))  # Fitness function
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2)  # Gaussian mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection

def run_ga():
    # Initialize population and tracking structures
    population = toolbox.population(n=50) # Population size
    hall_of_fame = tools.HallOfFame(1) # Track best individual
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])  # Track best fitness
    stats.register("min", np.min)  # Minimum fitness in population

    best_fitness_per_generation = []  # Track best fitness at each generation

    # Track best fitness at each generation
    def record_stats():
        fitness_values = [ind.fitness.values[0] for ind in population]
        best_fitness = min(fitness_values)
        best_fitness_per_generation.append(best_fitness)

    # Run the evolutionary algorithm
    for generation in range(100):
        population = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)  # Variation: crossover & mutation
        invalid_individuals = [ind for ind in population if not ind.fitness.valid]  # Invalid individuals
        fitnesses = map(toolbox.evaluate, invalid_individuals)  # Evaluate fitness
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit
        population = toolbox.select(population, len(population))  # Selection
        hall_of_fame.update(population)  # Update best solution
        record_stats()  # Record best fitness

    return best_fitness_per_generation  # Return best fitness history

if __name__ == "__main__":
    # Run GA and save convergence data
    fitness_history = run_ga()
    generations = np.arange(1, len(fitness_history) + 1)  # Generate generation indices for DataFrame

    # Create DataFrame from recorded fitness values
    df = pd.DataFrame({'Generation': generations, 'BestFitness': fitness_history})

    # Save convergence data to CSV
    df.to_csv("fitness_data/ga.csv", index=False) 

    # Plot convergence curve
    plt.plot(generations, fitness_history, label="GA")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Convergence on Rastrigin")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fitness_data/ga_plot.png")
    plt.show()
