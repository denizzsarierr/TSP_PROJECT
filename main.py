import pandas as pd
import math

class City:


    def __init__(self,id,x_coord,y_coord):
        self.id = int(id)
        self.x_coord = float(x_coord)
        self.y_coord = float(y_coord)
        
    def __str__(self):
        
        return f"ID : {self.id}, X : {self.x_coord}, Y : {self.y_coord}"


    def find_distance(self,compare_city):

        distance = ((self.x_coord - compare_city.x_coord)**2 + (self.y_coord - compare_city.y_coord)**2)**0.5

        return distance



class Parser:

    def __init__(self,file_path):

        self.dataset_info = {}
        self.city_data = []
        self.path = file_path
        self._convert_frame()

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
            
    def _convert_frame(self):

        self._parse_date()
        df = pd.DataFrame([{"id": c.id, "x_coordinate": c.x_coord, "y_coordinate": c.y_coord} for c in self.city_data])
        df.drop_duplicates(inplace = True)
        return df


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


if __name__ == "__main__":

    file_Name = 'berlin11_modified.tsp'

    data = Parser(file_path = file_Name)

    for d in data.city_data:
        for a in data.city_data:
            if d.id != a.id:
                print(f"Distance between City {d.id} and City {a.id} is {d.find_distance(a)}")

    