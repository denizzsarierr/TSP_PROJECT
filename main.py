import pandas as pd




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

    liste,diger = parse_data(file_Name)

    print(liste)
    print(diger)