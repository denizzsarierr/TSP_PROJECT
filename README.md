# Traveling Salesman Problem (TSP) Project

This project is a **university assignment** focused on solving the **Traveling Salesman Problem (TSP)** using **Object-Oriented Programming (OOP)** principles in Python.

At this stage, the project implements **data storage, parsing, and basic algorithmic components** of the TSP system — laying the foundation for future optimization and visualization.

## Objectives

- Model TSP using **OOP** with cities, coordinates, and distances.
- Parse `.tsp` files and store city data efficiently.
- Calculate and store **pairwise distances** in a DataFrame and distance matrix.
- Generate **random solutions** and compute **fitness** (total tour distance).
- Implement a **Greedy nearest-neighbor heuristic**.
- Implement a **Genetic Algorithm (GA)** with elitism, tournament selection, crossover, and mutations.
- Track best solutions and visualize results with **matplotlib**.

---

## Current Implementation

The project includes:

- **OOP design** for TSP components.
- Parsing `.tsp` files and creating `City` objects.
- Pairwise distance calculation stored in a **DataFrame** and a **distance matrix**.
- **Random solution generation** and fitness evaluation (total distance).
- **Greedy nearest-neighbor heuristic**.
- **Genetic Algorithm (GA)** with elitism, tournament selection, PMX crossover, multiple mutation strategies, and random immigrant injection.
- GA evolution with best solution tracking and optional **matplotlib** visualization.

### Implemented Classes

| Class        | Description                                                                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `City`       | Represents a city with attributes `id`, `x_coord`, `y_coord`, and distance list to other cities.                                                 |
| `Parser`     | Reads `.tsp` files, parses city coordinates, calculates pairwise distances, builds distance matrix, and returns a DataFrame.                     |
| `Solution`   | Stores candidate TSP solutions, calculates fitness, generates random solutions, and provides tour information.                                   |
| `Greedy`     | Implements a **nearest-neighbor heuristic** for TSP, with methods to run greedy tours from any starting city and find the best starting point.   |
| `Population` | Implements a **Genetic Algorithm population**, manages selection, crossover, mutation, epoch evolution, and tracks the best individual solution. |
