import pandas as pd
import math
import numpy as np

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
    
    def rand_solution(self):

        cities_test = [city.id for city in data.city_data]

        for _ in range(1,len(cities_test) + 1):
            random_sol = np.random.permutation(cities_test).tolist()
            
            if len(set(random_sol)) != len(cities_test):
                
                raise ValueError("Parsing failed.")
            
            self.solution.append(random_sol)
        
        print('Solution, Parsing successful.')
        return self.solution
    

class Solution:
    
    def __init__(self,data):

        self.solution_list = []
        self.data = data


    def rand_solution(self):
        
        cities_solved = [c.id for c in self.data.city_data]

        for _ in range(1,len(cities_solved) + 1):
            random_sol = np.random.permutation(cities_solved).tolist()
            
            if len(set(random_sol)) != len(cities_solved):
                
                raise ValueError("Parsing failed.")
            
            self.solution_list.append(random_sol)
        
        return self.solution_list
    
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
        
        print("Route:", " -> ".join(str(c) for c in solution))


def test_parsing(data):

    cities_test = [_.id for _ in data.city_data]
    print(f"Original: {cities_test}\n------------------" )

    for i in range(1,len(cities_test) + 1):
        random_sol = np.random.permutation(cities_test)
        
        if len(set(random_sol)) == len(cities_test):
            print(f"Test solution: {random_sol}")
            print(f"{i}. Solution, Parsing successful.")
    

        else:
            print("Parsing failed.")

if __name__ == "__main__":

    pd.set_option('display.max_rows', None)
    file_b11 = 'berlin11_modified.tsp'
    file_b52 = 'berlin52.tsp'
    
    data = Parser(file_path = file_b52)
    dataset_info = data.dataset_info
    distances = data.city_dataframe()
    
    # print(data.city_data)
    # random_solution = data.rand_solution()
    # print(random_solution)
    # data.rand_solution()
    # The First Task: Parse and convert the data into a DataFrame and test it.
    # res = distances.loc[(distances['FCID'] == 1) & (distances['SCID'] == 2)]
    # print(res)
    # print(dataset_info)
    # print(distances.head(10))
    # test_parsing(data)
    # ------------------------------------------------------------------------

    # print(distances)
    solution1 = Solution(data)
    solution_list = solution1.solution_list
    print(solution1.rand_solution())


    for i in solution_list:

        total = solution1.calculate_fitness(i)
        print(total)