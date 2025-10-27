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
    
# Basic parser without OOP.
def parse_data(file_name):

    dataset_info = {}
    city_data = []


    is_city_section = False

    with open(file_name,"r") as file:

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
                dataset_info[variable] = info
                continue

            # Parsing the coordinate and id of the city.
            details = line.split()

            id = details[0]
            x = details[1]
            y = details[2]
            
            city_data.append((id,x,y))


    city_data_pd = pd.DataFrame(city_data,columns=["id","x_coordinate","y_coordinate"])

    return dataset_info,city_data_pd

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
    
    # The First Task: Parse and convert the data into a DataFrame and test it.

    # print(dataset_info)
    # print(distances)
    # test_parsing(data)

    # ------------------------------------------------------------------------