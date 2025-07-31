import pandas as pd
import matplotlib.pyplot as plt

ga = pd.read_csv("data/ga_fitness.csv")
de = pd.read_csv("data/de_fitness.csv")
pso = pd.read_csv("data/pso_fitness.csv")

plt.plot(ga["Generation"], ga["BestFitness"], label="GA")
plt.plot(de["Generation"], de["BestFitness"], label="DE")
plt.plot(pso["Generation"], pso["BestFitness"], label="PSO")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Convergence Comparison on Rastrigin Function")
plt.legend()
plt.grid()
plt.savefig("data/convergence_comparison.png")
plt.show()