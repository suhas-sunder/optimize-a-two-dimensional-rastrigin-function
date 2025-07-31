import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO
from utils.rastrigin import rastrigin

def objective(X):
    return np.array([rastrigin(x) for x in X])

optimizer = GlobalBestPSO(n_particles=30, dimensions=2, options={'c1': 1.5, 'c2': 1.5, 'w': 0.7}, bounds=(np.array([-5.12, -5.12]), np.array([5.12, 5.12])))

cost, pos = optimizer.optimize(objective, iters=100)

history = optimizer.cost_history
pd.DataFrame({'Generation': np.arange(1, len(history)+1), 'BestFitness': history}).to_csv("data/pso_fitness.csv", index=False)

# Optional: plot
plt.plot(history, label="PSO")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("PSO Convergence on Rastrigin")
plt.grid()
plt.savefig("data/pso_plot.png")