import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution # Differential Evolution
from utils.shared_rastrigin_fcn import rastrigin  # Rastrigin function

# Initialize list to store best fitness per generation
best_fitness_history = []

# Callback function to record best fitness at each generation
def record_generation(xk, convergence=None):
    fitness = rastrigin(xk)
    best_fitness_history.append(fitness)
    print(f"Generation {len(best_fitness_history)} - Fitness: {fitness:.6e}")


# Define search space bounds for the 2D Rastrigin function
bounds = [(-5.12, 5.12), (-5.12, 5.12)]

# Run Differential Evolution optimization
result = differential_evolution(
    func=rastrigin,
    bounds=bounds,
    strategy='best1bin',       # Common DE strategy
    maxiter=100,               # Number of generations
    callback=record_generation,
    disp=True                  # Print progress in terminal
)

# Extract best fitness history
generation_indices = np.arange(1, len(best_fitness_history) + 1)

# Create DataFrame from recorded fitness values
df = pd.DataFrame({
    'Generation': generation_indices,
    'BestFitness': best_fitness_history
})

# Save convergence data to CSV
df.to_csv("fitness_data/de.csv", index=False)

# Plot convergence curve
plt.plot(df['Generation'], df['BestFitness'], label="DE")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("DE Convergence on Rastrigin")
plt.grid(True)
plt.tight_layout()
plt.savefig("fitness_data/de_plot.png")
plt.show()
