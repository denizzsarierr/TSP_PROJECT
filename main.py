import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import random

SEED = 41
random.seed(SEED)
np.random.seed(SEED)

class City:

    def __init__(self,id,x_coord,y_coord):
        self.id = int(id)
        self.x_coord = float(x_coord)
        self.y_coord = float(y_coord)
        self.distances = []
        
    def __str__(self):
        
        return f"ID : {self.id}, X : {self.x_coord}, Y : {self.y_coord}"

    # Finding distance between two cities. We use it in our Parser class.
    def find_distance(self,compare_city):

        distance = ((self.x_coord - compare_city.x_coord)**2 + (self.y_coord - compare_city.y_coord)**2)**0.5
        self.distances.append((compare_city.id,distance))

        return distance

class Parser:

    def __init__(self,file_path):

        self.dataset_info = {}
        self.city_data = []
        self.path = file_path
        self.distance_df = []
        self._convert_frame()
        self.solution = []
        self.dist_matrix = None
        # Call the parsing and converting function when the object is created.

    def _parse_date(self):

        with open(self.path,"r") as file:
            
            is_city_section = False 
            for line in file:

                line = line.strip()
                
                # Check if it is end of the dataset.
                if line.startswith('EOF'):
                    break

                # Check if it is alreay in the city section.
                if line.startswith('NODE_COORD_SECTION'):

                    is_city_section = True
                    continue
                
                # Parsing the informations that are not releated to city.
                if not is_city_section:

                    variable,info = line.split(":")
                    self.dataset_info[variable] = info
                    continue

                # Parsing the coordinate and id of the city.
                else: 
                
                    details = line.split()

                    city = City(details[0],details[1],details[2])
                
                    self.city_data.append(city)

    # Converting our list(data we got from tsp file) to a DataFrame.
    def _convert_frame(self):

        self._parse_date()
        df = pd.DataFrame([{"id": c.id, "x_coordinate": c.x_coord, "y_coordinate": c.y_coord} for c in self.city_data])
        df.drop_duplicates(inplace = True)
        return df

    def city_dataframe(self):

        # Here we use the find_distance method of City class to find distances between each city.
        temp_distances = []
        for c1 in self.city_data:
            for c2 in self.city_data:
                # To avoid duplicate
                if c2.id > c1.id:
                    distance = c1.find_distance(c2)
                    temp_distances.append((c1.id,c2.id,distance))
            
        self.distance_df = pd.DataFrame(temp_distances,columns=["FCID","SCID","Distance"])
        return self.distance_df
    
    def build_distance_matrix(self):
        n = len(self.city_data)
        self.dist_matrix = np.zeros((n+1, n+1))

        for _, row in self.distance_df.iterrows():
            a = int(row["FCID"])
            b = int(row["SCID"])
            d = float(row["Distance"])
            self.dist_matrix[a][b] = d
            self.dist_matrix[b][a] = d


class Solution:
    
    def __init__(self,data):

        self.solution_list = []
        self.data = data
        self.results = {}

    def rand_solution(self):
        
        cities_solved = [c.id for c in self.data.city_data]
        
        for _ in range(100):

            random_sol = np.random.permutation(cities_solved).tolist()
            
            if len(set(random_sol)) != len(cities_solved):
                
                raise ValueError("Parsing failed.")
            
            self.solution_list.append(random_sol)

        
        return self.solution_list
    
    def single_rand_solution(self):
        
        cities_solved = [c.id for c in self.data.city_data]
        
        random_sol = np.random.permutation(cities_solved).tolist()
                     
        return random_sol
    
    def calculate_fitness(self,solution):

        total_distance = 0
        distances_df = self.data.distance_df
        
        for i in range(len(solution)):
            
            distance = 0
            
            if i == len(solution) - 1:

                current = solution[i]
                next = solution[0]

            else:
                current = solution[i]
                next = solution[i + 1]

            # CHANGED CHANGED
            # distance = distances_df.loc[((distances_df["FCID"] == current) & (distances_df["SCID"] == next)) |
            #                          ((distances_df["FCID"] == next) & (distances_df["SCID"] == current)),
            #                            "Distance" ].values[0]
            distance = self.data.dist_matrix[current][next]
            total_distance += distance

        #self.info(solution=solution)
        
        return total_distance

    def info(self,solution):
        
        return "Route:" + " -> ".join(str(c) for c in solution)


class Greedy:

    def __init__(self,data):

        self.data = data
        self.route = []
        
    def greedy_solution(self,start_point):
        distance_df = self.data.distance_df

        total_distance = 0      
        current_city_id = start_point
        next_city_id = 0
        visited_cities = [start_point]
        self.route = [start_point]
        
        while len(visited_cities) < len(self.data.city_data):
            
            point_rows = distance_df[
            (distance_df["FCID"] == current_city_id) | (distance_df["SCID"] == current_city_id)]
            point_rows = point_rows.sort_values("Distance")
            best_min = float('inf')

            for _, row in point_rows.iterrows():
                
                if row["FCID"] == current_city_id:
                    next_id = int(row["SCID"])
                else:
                    next_id = int(row["FCID"])
                
                distance = row["Distance"]

                
                if next_id not in visited_cities and distance < best_min:
                    
                    best_min = distance
                    next_city_id = next_id
                    # print(best_min)

            self.route.append(next_city_id)
            visited_cities.append(next_city_id)
            total_distance += best_min            
            current_city_id = int(next_city_id)
        
        returning_dist = distance_df.loc[
        ((distance_df["FCID"] == current_city_id) & (distance_df["SCID"] == start_point)) |
        ((distance_df["FCID"] == start_point) & (distance_df["SCID"] == current_city_id)),
        "Distance"
        ].values[0]

        total_distance += returning_dist
        
        return total_distance,self.route
    
    def info(self,solution):
        
        return "Route: " + " -> ".join(str(c) for c in solution)
    
    def greedy_for_each(self):

        results = {}
        
        for i in data.city_data:

            dist,route = self.greedy_solution(i.id)
            results[i.id] = (int(dist),route)

        return results

class Population:

    def __init__(self,data):
        self.data = data
        self.population = []
        self.winners = []
        self.fitness_cal = Solution(self.data)
        self.best_individual = None

    def create_population(self,individuals,greedies = 0):
        
        self.population = []
        
        solution_picker = Solution(self.data)

        if greedies > 0:
            
            greedy_picker = Greedy(self.data)

            all_greedy = list(greedy_picker.greedy_for_each().values())

            sorted_greedy_dict = sorted(all_greedy, key=lambda x: x[0])

            greedy_routes = [route for (_, route) in sorted_greedy_dict[:greedies]]

            for i in greedy_routes:

                fitness = solution_picker.calculate_fitness(i)
                self.population.append({"route": i, "fitness": int(fitness)})
            
        while len(self.population) < individuals:
            
            route = solution_picker.single_rand_solution()
            fitness = solution_picker.calculate_fitness(route)

            if not any(x["route"] == route for x in self.population):

                self.population.append({"route": route, "fitness": int(fitness)})

        return self.population
    
    def info(self):
        
        if not self.population:
            print("Population is empty.")
            return

        fitness_values = [x["fitness"] for x in self.population]

        min_dist = np.argmin(fitness_values)   # smaller distance is better
        max_dist = np.argmax(fitness_values)

        print("Population Information:")
        print("-----------------------")
        print(f"Size of population: {len(self.population)}")
        print(f"Best fitness (shortest distance): {fitness_values[min_dist]}")
        print(f"Best route: {self.population[min_dist]['route']}")
        print(f"Median fitness: {np.median(fitness_values)}")
        print(f"Worst fitness (longest distance): {fitness_values[max_dist]}")
        print(f"Average fitness: {np.mean(fitness_values):.2f}")
    
    def selections_combined(self,tournament_size,tournament_count,elite_count):
        
        # Elitisism
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'])

        elites = sorted_pop[:elite_count]

        self.winners = elites.copy()

        # -----Tournament Selection-----
        for _ in range(tournament_count):
            
            random_routes = [random.choice(self.population) for _ in range(tournament_size)]
            
            winner = min(random_routes, key=lambda x: x["fitness"])

            self.winners.append(winner)
        
        return self.winners
    def pmx_crossover(self, parent1, parent2):
        size = len(parent1)
        child = [None]*size
        start, end = sorted(random.sample(range(size), 2))

       
        child[start:end+1] = parent1[start:end+1]

       
        for i in range(start, end+1):
            if parent2[i] not in child:
                val = parent2[i]
                idx = i
                while child[idx] is not None:
                    idx = parent2.index(parent1[idx])
                child[idx] = val

       
        for i in range(size):
            if child[i] is None:
                child[i] = parent2[i]

        return child
    def crossover(self,pairs_count = 0):
        
        crossed_gen = []

        for _ in range(pairs_count):
            parent1, parent2 = random.sample(self.winners, 2)  # picks 2 distinct parents
            child1 = self.pmx_crossover(parent1["route"], parent2["route"])
            child2 = self.pmx_crossover(parent2["route"], parent1["route"])
            crossed_gen.append({"route": child1,
                                "fitness": int(self.fitness_cal.calculate_fitness(child1))})
            crossed_gen.append({"route": child2,
                                "fitness": int(self.fitness_cal.calculate_fitness(child2))})

        return crossed_gen
    
    def mutation(self,ind,mut_rate):
        
        route = ind['route'][:]
        if random.random() < mut_rate:
            choice = random.random()
            if choice < 0.33:
                # Swap mutation
                i, j = random.sample(range(len(route)), 2)
                route[i], route[j] = route[j], route[i]
            elif choice < 0.66:
                # Inversion mutation
                i, j = sorted(random.sample(range(len(route)), 2))
                route[i:j+1] = route[i:j+1][::-1]
            else:
                # Scramble mutation
                i, j = sorted(random.sample(range(len(route)), 2))
                segment = route[i:j+1]
                random.shuffle(segment)
                route[i:j+1] = segment
        
        fitness = int(self.fitness_cal.calculate_fitness(route))
        return {"route": route, "fitness": fitness}
    
    def create_epoch(self, tournament_size, tournament_count, elite_count, mut_rate):

        self.selections_combined(tournament_size=tournament_size,
                                 tournament_count=tournament_count,
                                 elite_count=elite_count)
        population_size = len(self.population)
        needed_children = population_size - elite_count  # Number of children to produce
        pairs_count = needed_children // 2
        
        crossed_gen = self.crossover(pairs_count=pairs_count)

        mutated_gen = []
        
        for i in crossed_gen:

            mutated = self.mutation(ind = i, mut_rate=mut_rate)
            mutated_gen.append(mutated)
        
        if needed_children % 2 == 1:
            parent = random.choice(self.winners)
            mutated_gen.append(self.mutation(parent, mut_rate))

        
        
        sorted_old = sorted(self.population, key=lambda x: x['fitness'])
        elites = sorted_old[:elite_count]
        
        new_population = elites + mutated_gen[:needed_children]
        self.population = new_population

        # --- 8. Track global best (unchanged) ---
        current_best = min(self.population, key=lambda x: x["fitness"])
        if self.best_individual is None or current_best["fitness"] < self.best_individual["fitness"]:
            self.best_individual = current_best.copy()

        return self.best_individual

if __name__ == "__main__":

    pd.set_option('display.max_rows', None)
    file_b11 = 'berlin11_modified.tsp'
    file_b52 = 'berlin52.tsp'
    file_kroA = 'kroA100.tsp'
    file_kroA150 = 'kroA150.tsp'
    # 21886
    data = Parser(file_path=file_b52)
    dataset_info = data.dataset_info
    data.city_dataframe()
    data.build_distance_matrix()

    # ------------------- Parameter sets to test -------------------
    parameter_sets = [
        {"population_size": 100, "mutation_rate": 0.05, "greedies": 0},
        {"population_size": 200, "mutation_rate": 0.15, "greedies": 20},
        {"population_size": 500, "mutation_rate": 0.25, "greedies": 40}
    ]

    num_epochs = 500      # can reduce for testing
    tournament_size = 5       
    tournament_count = 15     
    elite_count = 2           

    all_experiments = []

    # Track globally best solution across all parameter sets
    global_best_fitness = float('inf')
    global_best_route = []

    for params in parameter_sets:
        population_size = params["population_size"]
        mutation_rate = params["mutation_rate"]
        greedies = params["greedies"]

        print(f"\nRunning GA with Population={population_size}, Mutation={mutation_rate}, Greedy={greedies}")

        population = Population(data)
        population.create_population(individuals=population_size, greedies=greedies)

        best_fitness_list = []

        for epoch in range(num_epochs):
            best_individual = population.create_epoch(
                tournament_size=tournament_size,
                tournament_count=tournament_count,
                elite_count=elite_count,
                mut_rate=mutation_rate,
            )
            best_fitness_list.append(best_individual['fitness'])

        print(f"Best distance: {population.best_individual['fitness']}")

        
        if population.best_individual['fitness'] < global_best_fitness:
            global_best_fitness = population.best_individual['fitness']
            global_best_route = population.best_individual['route']

        # Store results
        all_experiments.append({
            "Population": population_size,
            "Mutation": mutation_rate,
            "Greedy": greedies,
            "BestFitness": population.best_individual['fitness'],
            "BestFitnessOverTime": best_fitness_list
        })

    # ------------------- Plot comparison -------------------
    plt.figure(figsize=(10, 6))
    for exp in all_experiments:
        label = f"Pop={exp['Population']}, Mut={exp['Mutation']}, Greedy={exp['Greedy']}"
        plt.plot(range(1, num_epochs+1), exp["BestFitnessOverTime"], label=label)

    plt.title("GA Fitness Progression for Different Parameter Sets")
    plt.xlabel("Epoch")
    plt.ylabel("Best Fitness (Distance)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ------------------- Plot Best TSP Solution on Map with Parameters -------------------
    x_coords = [data.city_data[city_id - 1].x_coord for city_id in global_best_route]
    y_coords = [data.city_data[city_id - 1].y_coord for city_id in global_best_route]

    # Close the loop
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    plt.figure(figsize=(10, 8))

    # Plot the route
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue', linewidth=2, markersize=6)

    # Highlight the starting city with a red star
    start_city = data.city_data[global_best_route[0] - 1]
    plt.scatter(start_city.x_coord, start_city.y_coord, color='red', s=100, marker='*', label='Start City')

    # Annotate city IDs
    for i, city_id in enumerate(global_best_route):
        city = data.city_data[city_id - 1]
        plt.text(city.x_coord + 1, city.y_coord + 1, str(city.id), fontsize=9)

    # Find which parameter set produced this best solution
    best_exp = None
    for exp in all_experiments:
        if exp['BestFitness'] == global_best_fitness:
            best_exp = exp
            break

    # Add parameters as a title
    if best_exp is not None:
        plt.title(
            f"Best TSP Solution (Distance={global_best_fitness})\n"
            f"Population={best_exp['Population']}, Mutation={best_exp['Mutation']}, Greedy={best_exp['Greedy']}"
        )
    else:
        plt.title(f"Best TSP Solution (Distance={global_best_fitness})")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()
    