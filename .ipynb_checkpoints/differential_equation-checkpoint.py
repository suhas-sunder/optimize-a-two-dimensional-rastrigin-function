import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from utils.rastrigin import rastrigin

history = []

def callback(xk, convergence):
    history.append(rastrigin(xk))

bounds = [(-5.12, 5.12), (-5.12, 5.12)]
result = differential_evolution(rastrigin, bounds, strategy='best1bin', maxiter=100, callback=callback)

# Save fitness per generation
df = pd.DataFrame({'Generation': np.arange(1, len(history)+1), 'BestFitness': history})
df.to_csv("data/de_fitness.csv", index=False)

# Optional plot
plt.plot(history, label="DE")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("DE Convergence on Rastrigin")
plt.grid()
plt.savefig("data/de_plot.png")
