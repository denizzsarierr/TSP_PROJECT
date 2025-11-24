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


            distance = distances_df.loc[((distances_df["FCID"] == current) & (distances_df["SCID"] == next)) |
                                     ((distances_df["FCID"] == next) & (distances_df["SCID"] == current)),
                                       "Distance" ].values[0]

            total_distance += distance

        self.info(solution=solution)
        
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
            
            print(f"\nStarting city: {i.id}")
            print(self.info(route))
            print(f"Total distance: {int(dist)}")

        best_start_point, (best_distance, best_route) = min(results.items(),key = lambda x : x[1][0]) 

        print(f"\nBest Starting Point: {best_start_point}")
        print(self.info(best_route))
        print(f"Total distance: {int(best_distance)}")

        return results

class Population:

    def __init__(self,data):
        self.data = data
        self.population = []

    def create_population(self,individuals,greedies = 0):
        
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
        winners = []

        sorted_pop = sorted(self.population, key=lambda x: x['fitness'])

        elites = sorted_pop[:elite_count]

        winners = elites.copy()


        # -----Tournament Selection-----
        for _ in range(tournament_count):
            
            random_routes = [random.choice(self.population) for _ in range(tournament_size)]
            
            winner = min(random_routes, key=lambda x: x["fitness"])

            winners.append(winner)
        


        # -----Roulette Selection-----
        weights = []

        for ind in self.population:

            weight = 1 / ind["fitness"]
            weights.append(weight)

        # Total weight
        total_w = sum(weights)    
        
        
        # Probabilities 
        probabilities = []
        for weight in weights:

            probability = weight / total_w
            probabilities.append(probability)

        for _ in range(len(self.population) - len(winners)):

            rand = random.random()

            c = 0

            for ind,prob in zip(self.population,probabilities):

                c += prob

                if rand <= c:
                    winners.append(ind)
                    break



        return winners
        



if __name__ == "__main__":

    pd.set_option('display.max_rows', None)
    file_b11 = 'berlin11_modified.tsp'
    file_b52 = 'berlin52.tsp'
    
    data = Parser(file_path = file_b11)
    dataset_info = data.dataset_info
    data.city_dataframe()
    
    # print(data.city_data)
    # The First Task: Parse and convert the data into a DataFrame and test it.
    # test_parsing(data)

    # ------------------------TASK 2-------------------------------------

    solution1 = Solution(data)
    solution_list = solution1.rand_solution()

    random_results = []

    for i in solution_list:
        
        
        total = solution1.calculate_fitness(i)
        random_results.append((i,int(total)))
        print(solution1.info(i), "Total distance:", total)
        #solution1.plot_solution(i)

    best_random_route , best_random_distance = min(random_results, key=lambda x: x[1]) 

    greed = Greedy(data=data)
    greedy_results = greed.greedy_for_each()
    
    best_greedy_distance = min(greedy_results.values(), key=lambda x: x[0])[0]
    best_greedy_route = min(greedy_results.values(), key=lambda x: x[0])[1]
    
    if best_random_distance < best_greedy_distance:
        
        print(f"Random solution is better than greedy. Random: {best_random_distance}, Greedy: {best_greedy_distance}")
        print(f"Route: {best_random_route}")

    elif best_random_distance > best_greedy_distance:

        print(f"Greedy solution is better than random. Random: {best_random_distance}, Greedy: {best_greedy_distance}")
        print(f"Route: {best_greedy_route}")
    else:
        print(f"Random and Greedy are equal. Random: {best_random_distance}, Greedy: {best_greedy_distance}")
        print(f"Routes: Random - {best_random_route}, Greedy - {best_greedy_route}")
    
    # ------------------------TASK 2-------------------------------------   
 
    # ------------------------TASK 3-------------------------------------   
    population = Population(data)

    new_pop = population.create_population(individuals=50,greedies=2)
    print('-------------')
    print(new_pop)
    print('-------------')
    print(population.info())
    a = population.selections_combined(tournament_size=2,tournament_count=20,elite_count=3)
    print(a)
    