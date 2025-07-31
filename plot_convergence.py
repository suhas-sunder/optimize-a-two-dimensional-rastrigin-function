import pandas as pd
import matplotlib.pyplot as plt

# Load fitness data from CSVs
ga = pd.read_csv("fitness_data/ga.csv")
de = pd.read_csv("fitness_data/de.csv")
pso = pd.read_csv("fitness_data/pso.csv")

# Plot convergence on log scale
plt.figure(figsize=(10, 6))
plt.plot(ga["Generation"], ga["BestFitness"], label="GA", linewidth=2)
plt.plot(de["Generation"], de["BestFitness"], label="DE", linewidth=2)
plt.plot(pso["Generation"], pso["BestFitness"], label="PSO", linewidth=2)

plt.xlabel("Generation")
plt.ylabel("Best Fitness (log scale)")
plt.title("Convergence Comparison on Rastrigin Function")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fitness_data/convergence_comparison_log.png")
plt.show()
