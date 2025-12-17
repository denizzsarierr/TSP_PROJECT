import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import random

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

    # AI GENERATED MATPLOTLIB
    # def plot_solution(self, solution):
    #     df = pd.DataFrame([{"id": c.id, "x": c.x_coord, "y": c.y_coord} for c in self.data.city_data])

    #     # Reorder by solution order
    #     ordered = df.set_index("id").loc[solution]

    #     # Close the loop (add the first city again at the end)
    #     ordered = pd.concat([ordered, ordered.iloc[[0]]])

    #     # Plot
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(ordered["x"], ordered["y"], marker="o", linestyle="-")
    #     for i, row in ordered.iterrows():
    #         plt.text(row["x"] + 10, row["y"] + 10, str(i), fontsize=9)
    #     plt.title("TSP Route Visualization")
    #     plt.xlabel("X Coordinate")
    #     plt.ylabel("Y Coordinate")
    #     plt.grid(True)
    #     plt.show()

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
            
            #print(f"\nStarting city: {i.id}")
            #print(self.info(route))
            #print(f"Total distance: {int(dist)}")

        #best_start_point, (best_distance, best_route) = min(results.items(),key = lambda x : x[1][0]) 

        #print(f"\nBest Starting Point: {best_start_point}")
        #print(self.info(best_route))
        #print(f"Total distance: {int(best_distance)}")

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

        # Copy the segment from parent1
        child[start:end+1] = parent1[start:end+1]

        # Fill remaining positions using mapping
        for i in range(start, end+1):
            if parent2[i] not in child:
                val = parent2[i]
                idx = i
                while child[idx] is not None:
                    idx = parent2.index(parent1[idx])
                child[idx] = val

        # Fill the rest of the positions
        for i in range(size):
            if child[i] is None:
                child[i] = parent2[i]

        return child
    def crossover(self,pairs_count = 0):
        
        crossed_gen = []

        pairs = [tuple(random.sample(self.winners, 2)) for _ in range(pairs_count)]
        for parent1,parent2 in pairs:
            
            # r_length = len(parent1["route"])
            # start,end = sorted(random.sample(range(r_length), 2))
            
            # crossed_child = [None] * r_length

            # crossed_child[start:end + 1] = parent1["route"][start:end + 1]
            
            # index = (end + 1) % r_length
            # for c in parent2["route"]:
            #     if c not in crossed_child:
            #         crossed_child[index] = c
            #         index = (index + 1) % r_length

            # fitness = int(self.fitness_cal.calculate_fitness(crossed_child))
            # crossed_gen.append({"route" : crossed_child, "fitness" : fitness})
            child1 = self.pmx_crossover(parent1["route"], parent2["route"])
            child2 = self.pmx_crossover(parent2["route"], parent1["route"])
            crossed_gen.append({"route": child1, "fitness": int(self.fitness_cal.calculate_fitness(child1))})
            crossed_gen.append({"route": child2, "fitness": int(self.fitness_cal.calculate_fitness(child2))})
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

        # rand_number = random.random()

        # if rand_number < mut_rate:

        #     i = random.randint(0,len(route) - 1)
        #     y = random.randint(0,len(route) - 1)

        #     while y == i:
        #         y = random.randint(0,len(route) - 1)
                
        #     temp_route = route[i:y + 1]

        #     reverse_temp_route = temp_route[::-1]

        #     route[i:y + 1] = reverse_temp_route
        

        fitness = int(self.fitness_cal.calculate_fitness(route))
        return {"route": route, "fitness": fitness}
    
    def create_epoch(self, tournament_size, tournament_count, elite_count, mut_rate, pairs_count):

        self.selections_combined(tournament_size=tournament_size,
                                 tournament_count=tournament_count,
                                 elite_count=elite_count)
        
        crossed_gen = self.crossover(pairs_count=pairs_count)

        mutated_gen = []

        for i in crossed_gen:

            mutated = self.mutation(ind = i, mut_rate=mut_rate)
            mutated_gen.append(mutated)
        
        sorted_old = sorted(self.population, key=lambda x: x['fitness'])
        elites = sorted_old[:elite_count]
        
        new_population = elites + mutated_gen


        # NEW PART NEW PART NEW PART
        num_random = max(1, int(0.1 * len(self.population)))  # 10% of population
        for _ in range(num_random):
            rand_sol = self.fitness_cal.single_rand_solution()
            rand_fitness = int(self.fitness_cal.calculate_fitness(rand_sol))
            # Replace worst individuals in the population with random ones
            new_population[-(_+1)] = {"route": rand_sol, "fitness": rand_fitness}



        while len(new_population) < len(self.population):
            rand_sol = self.fitness_cal.single_rand_solution()
            new_population.append({
                "route": rand_sol,
                "fitness": int(self.fitness_cal.calculate_fitness(rand_sol))
            })

        self.population = new_population

        current_best = min(self.population, key=lambda x: x['fitness'])
        if self.best_individual is None or current_best['fitness'] < self.best_individual['fitness']:
            self.best_individual = current_best.copy()

        return self.best_individual


if __name__ == "__main__":

    pd.set_option('display.max_rows', None)
    file_b11 = 'berlin11_modified.tsp'
    file_b52 = 'berlin52.tsp'
    file_kroA = 'kroA100.tsp'
    
    data = Parser(file_path = file_b52)
    dataset_info = data.dataset_info
    data.city_dataframe()
    data.build_distance_matrix()

    population = Population(data)
    
    # ------------------- GA PARAMETERS -------------------
    num_epochs = 1000      
    tournament_size = 5       
    tournament_count = 15     
    elite_count = 1           
    mutation_rate = 0.15     
    population_size = 150
    #pairs_count = (population_size - elite_count) // 2 
    pairs_count = population_size
    greedies = 20

    # Create initial population
    population.create_population(individuals=population_size, greedies=greedies)

    # Track best fitness per epoch
    best_fitness_list = []

    for epoch in range(num_epochs):
        best_individual = population.create_epoch(
            tournament_size=tournament_size,
            tournament_count=tournament_count,
            elite_count=elite_count,
            mut_rate=mutation_rate,
            pairs_count=pairs_count
        )
        
        best_fitness_list.append(best_individual['fitness'])
    
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Best distance so far = {best_individual['fitness']}")
    
    # Final best solution
    print("\n=== FINAL BEST SOLUTION ===")
    print(f"Best distance: {population.best_individual['fitness']}")
    print(f"Best route: {population.best_individual['route']}")

    # Plot progress over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), best_fitness_list, marker='o', linestyle='-')
    plt.title("Genetic Algorithm Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Best Fitness (Distance)")
    plt.grid(True)
    plt.show()
