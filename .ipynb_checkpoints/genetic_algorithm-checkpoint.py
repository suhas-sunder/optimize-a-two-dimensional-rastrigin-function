import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from utils.rastrigin import rastrigin

random.seed(42)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5.12, 5.12)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", lambda x: (rastrigin(np.array(x)),))
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)

    fitness_history = []

    def record_stats(gen):
        fits = [ind.fitness.values[0] for ind in pop]
        fitness_history.append(min(fits))

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100,
                        stats=stats, halloffame=hof, verbose=False)

    for gen in range(100):
        record_stats(gen)

    return fitness_history

if __name__ == "__main__":
    fitness_history = run_ga()
    pd.DataFrame({'Generation': np.arange(1, 101), 'BestFitness': fitness_history}).to_csv("data/ga_fitness.csv", index=False)

    plt.plot(fitness_history, label="GA")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Convergence on Rastrigin")
    plt.grid()
    plt.savefig("data/ga_plot.png")