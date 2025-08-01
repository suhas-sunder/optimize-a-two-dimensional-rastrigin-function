import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO # Particle Swarm Optimization
from utils.shared_rastrigin_fcn import rastrigin # Rastrigin function

# Objective function wrapper for PSO (accepts matrix of particles)
def objective_function(particles):
    return np.array([rastrigin(particle) for particle in particles])

# PSO hyperparameters
swarm_size = 30 # Number of particles in the swarm
dimensions = 2 # 2D Rastrigin function
inertia_weight = 0.7 # Weight for inertia
cognitive_coeff = 1.5  # c1: self-confidence
social_coeff = 1.5     # c2: swarm influence

# Define search bounds for x and y in Rastrigin
lower_bounds = np.array([-5.12, -5.12])
upper_bounds = np.array([5.12, 5.12]) 
bounds = (lower_bounds, upper_bounds)

# Initialize the PSO optimizer
optimizer = GlobalBestPSO(
    n_particles=swarm_size,
    dimensions=dimensions,
    options={'c1': cognitive_coeff, 'c2': social_coeff, 'w': inertia_weight},
    bounds=bounds
)

# Run optimization
best_cost, best_position = optimizer.optimize(objective_function, iters=100)

# Extract fitness history
fitness_per_generation = optimizer.cost_history

# Create DataFrame from recorded fitness values
df = pd.DataFrame({
    'Generation': np.arange(1, len(fitness_per_generation) + 1),
    'BestFitness': fitness_per_generation
})

# Save to CSV
df.to_csv("fitness_data/pso.csv", index=False)

# Plot convergence
plt.plot(df['Generation'], df['BestFitness'], label="PSO")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("PSO Convergence on Rastrigin")
plt.grid(True)
plt.tight_layout()
plt.savefig("fitness_data/pso_plot.png")
plt.show()
