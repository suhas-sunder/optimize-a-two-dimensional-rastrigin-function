# ENGR 5010G - Advanced Optimisation Assignment Two: Rastrigin Function Optimization using GA, DE, and PSO

## Ontario Tech University - Faculty of Engineering and Applied Science

### Instructor: Md Asif Khan, PhD

### Term: Summer 2025

<img width="4188" height="1327" alt="image" src="https://github.com/user-attachments/assets/ac0b3d0d-f8c3-4f88-bfc1-63f2ec92b3c2" />

This project compares three population-based metaheuristic algorithms—Genetic Algorithm (GA), Differential Evolution (DE), and Particle Swarm Optimization (PSO)—for optimizing the 2D Rastrigin function:

\[
f(x, y) = 20 + x^2 + y^2 - 10[\cos(2\pi x) + \cos(2\pi y)]
\]

The goal is to locate the global minimum at (0, 0), where f(0, 0) = 0, and analyze the convergence behavior of each method.

## Algorithms

- **Genetic Algorithm (GA)** using `deap`
- **Differential Evolution (DE)** using `scipy.optimize.differential_evolution`
- **Particle Swarm Optimization (PSO)** using `pyswarms`

## Requirements

Install dependencies with:

```bash
pip install numpy matplotlib pandas pyswarms deap scipy
```

## Usage

Run the scripts:

```bash
python genetic_algorithm.py      # Runs Genetic Algorithm
python differential_equation.py      # Runs Differential Evolution
python particle_swarm_optimization.py     # Runs Particle Swarm Optimization
python plot_convergence.py  # Plots convergence of all methods
```

Raw data and plots can be found in the `fitness_data` folder.
