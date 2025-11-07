# Traveling Salesman Problem (TSP) Project

This project is a **university assignment** focused on solving the **Traveling Salesman Problem (TSP)** using **Object-Oriented Programming (OOP)** principles in Python.

At this stage, the project implements **data storage, parsing, and basic algorithmic components** of the TSP system — laying the foundation for future optimization and visualization.

## Objectives

- Understand the data structure and representation of the **Traveling Salesman Problem (TSP)**.
- Use **OOP design** to model cities, coordinates, and distances.
- Implement **robust data parsing** from `.tsp` files.
- Calculate distances between cities and store them in a **DataFrame** for efficient access.
- Implement **random solutions** and calculate their **fitness** (total travel distance).
- Implement a **Greedy algorithm** as a basic heuristic for TSP solutions.
- Visualize results and routes (optional, using `matplotlib`).

---

## Current Implementation

At this stage, the project includes:

- **Class-based design** for TSP problem components.
- **Parsing logic** that reads city information from `.tsp` files and constructs `City` objects.
- **Distance calculation** between every pair of cities, stored in a **Pandas DataFrame**.
- **Random solution generator** to create candidate tours.
- **Fitness function** to calculate the total distance of a tour.
- **Greedy algorithm** to generate tours starting from any city and choosing the nearest unvisited city iteratively.
- Optional **matplotlib visualization** for plotting routes.

### Implemented Classes

| Class     | Description                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------- |
| `City`    | Represents a city with attributes `id`, `x_coord`, `y_coord`, and distance list to other cities.  |
| `Parser`  | Reads `.tsp` files, parses city coordinates, calculates pairwise distances, and returns a DataFrame. |
| `Solution`| Stores candidate TSP solutions, calculates fitness, and provides tour information.                |
| `Greedy`  | Implements a **nearest-neighbor heuristic** for TSP, with methods to run greedy tours from any starting city and find the best starting point. |

---

