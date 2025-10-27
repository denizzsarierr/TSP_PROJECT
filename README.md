# Traveling Salesman Problem (TSP) Project

This project is a **university assignment** focused on solving the **Traveling Salesman Problem (TSP)** using **Object-Oriented Programming (OOP)** principles in Python.  

At this stage, the project implements the **data storage and parsing components** of the TSP system — preparing the foundation for later algorithmic development.

## Objectives

- Understand the data structure and representation of the **Traveling Salesman Problem (TSP)**.  
- Use **OOP design** to model cities, coordinates, and distances.  
- Implement robust **data parsing and validation** from input files.  
- Build a scalable base for future development (TSP algorithms, visualizations, etc.).  

---

##  Current Implementation

At this stage, the project includes:

- **Class-based design** for data handling and structure.  
- **Parsing logic** that reads city information (e.g., coordinates or adjacency matrices) from files.  
- **In-memory storage** of cities and distances for later algorithmic use.  
- **Error handling** for invalid or missing input data.  

### Implemented Classes

| Class | Description |
|--------|--------------|
| `City` | Represents a city with attributes like name, coordinates, and ID. |
| `Parser` | Reads and parses input files (e.g., `.txt`, `.csv`) and constructs `City` objects. |
