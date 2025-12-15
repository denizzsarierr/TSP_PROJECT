# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:25:46 2024

@author: smdyc
"""

import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt


file = 'berlin52.tsp'

# params


def parser_info(file):
    file_data = {}
    with open(file, 'r') as file:
        for x in file:
            splitted_line = x.split()
            if len(splitted_line) > 0:
                if 'NAME' in splitted_line[0]:
                    file_data['Name'] = ' '.join(splitted_line[1:])
                elif 'TYPE' in splitted_line[0]:
                    file_data['file_type'] = splitted_line[-1]
                elif 'COMMENT' in splitted_line[0]:
                    file_data['comment'] = ' '.join(splitted_line[1:])
                elif 'DIMENSION' in splitted_line[0]:
                    file_data['dimension'] = ' '.join(splitted_line[1:])
                elif 'EDGE_WEIGHT_TYPE' in splitted_line[0]:
                    file_data['weight_type'] = ' '.join(splitted_line[1:])
    return file_data


def parser_tsp(file):
    i = 1
    indexes = []
    data = {}
    data_x = []
    data_y = []
    file = open(file)
    for x in file:
        splitted_line = x.split()
        try:
            while len(splitted_line) > 0 and float(splitted_line[0]) == i:
                data_x.append(float(splitted_line[1]))
                data_y.append(float(splitted_line[2]))
                i += 1
                indexes.append(float(splitted_line[0]))
        except (ValueError, IndexError):
            continue

    data = {'city_number': indexes, 'x': data_x, 'y': data_y}
    return data


def city_getter(df, city_number):
    city = df[df['city_number'] == city_number]
    return city['x'].values[0], city['y'].values[0]


def find_distance(x1, y1, x2, y2):
    return (math.sqrt(((x2-x1)**2) + ((y2-y1)**2)))


def find_distance_loc(df, c1, c2):
    x1 = (city_getter(df, c1))[0]
    y1 = (city_getter(df, c1))[1]
    x2 = (city_getter(df, c2))[0]
    y2 = (city_getter(df, c2))[1]
    return (math.sqrt(((x2-x1)**2) + ((y2-y1)**2)))


def randomize_df(df):
    #random_df.reset_index(drop=True)
    tmp = df.sample(frac=1)
    return tmp.reset_index(drop=True)


def check_fitness(df_cities):
    total = 0
    for c in range(len(df_cities)):
        next_city = (c + 1) % len(df_cities)
        total += find_distance(df_cities['x'][c],
                               df_cities['y'][c],
                               df_cities['x'][next_city],
                               df_cities['y'][next_city])
    return total


def info(df):
    return df['city_number'].values


def info_parrent(parrent_df):
    return parrent_df.values[0][0]['city_number'].values


def greedy_alg(df, start):
    if start <= 0:
        raise Exception('Starting city can not be less than 1')

    visited_cities = []
    # List of all city numbers
    unvisited_cities = list(df['city_number'].values)
    current_city = unvisited_cities.pop(start-1)  # Start with the first city
    visited_cities.append(current_city)

    total_distance = 0

    while unvisited_cities:
        # Find the nearest city
        min_distance = float('inf')
        nearest_city = None

        for city in unvisited_cities:
            distance = find_distance_loc(df, current_city, city)
            if distance < min_distance:
                min_distance = distance
                nearest_city = city

        # Move to the nearest city
        total_distance += min_distance
        current_city = nearest_city
        visited_cities.append(current_city)
        unvisited_cities.remove(current_city)

    # Add distance back to the starting city to complete the cycle
    total_distance += find_distance_loc(df,
                                        visited_cities[-1], visited_cities[0])

    # Save the visited cities in DataFrame format
    visited_df = pd.DataFrame({
        'city_number': visited_cities,
        'x': [city_getter(df, city)[0] for city in visited_cities],
        'y': [city_getter(df, city)[1] for city in visited_cities]
    })

    return visited_df


def solution_saver(df, list_solution):
    output = pd.DataFrame({
        'city_number': list_solution,
        'x': [city_getter(df, city)[0] for city in list_solution],
        'y': [city_getter(df, city)[1] for city in list_solution]
    })
    return output


def initial_population(df, n_solutions, greedy_fraction=0.3):
    list_solutions = []
    list_fitness = []
    list_method = []

    n_greedy = int(n_solutions * greedy_fraction)

    # greedy part
    for _ in range(n_greedy):
        solution = greedy_alg(df, random.randint(1, len(df)))
        list_solutions.append(solution)
        list_fitness.append(check_fitness(solution))
        list_method.append('G')

    # random part
    for _ in range(n_solutions - n_greedy):
        solution = randomize_df(df)
        list_solutions.append(solution)
        list_fitness.append(check_fitness(solution))
        list_method.append('R')

    population = {'solution': list_solutions, 'fitness': list_fitness,
                  'method': list_method}

    return pd.DataFrame(population)


def population_info(population_df):
    print(f"Population Size: {len(population_df)}")
    print(f"Best Fitness: {population_df['fitness'].min()}")
    print(f"Median Fitness: {population_df['fitness'].median()}")
    print(f"Worst Fitness: {population_df['fitness'].max()}")


def tournament_selection(population_df, tournament_size):
    """
    Perform tournament selection on the population.
    """
    participants = population_df.sample(n=tournament_size)
    winner = participants.loc[participants['fitness'].idxmin()]
    output = {'solution': [winner.iloc[0]],
              'fitness': [winner.iloc[1]],
              'method': [winner.iloc[2]]}
    return pd.DataFrame(output)


def pmx_crossover(parrent1, parrent2,pmx_prob=0.9,df=file):
    """
    Perform Partially Matched Crossover (PMX) on two parents.
    """
    solution1, solution2 = info_parrent(
        parrent1).tolist(), info_parrent(parrent2).tolist()
    if random.random() > pmx_prob: 
        return solution_saver(df, solution1)
    size = len(solution1)
    pt1, pt2 = sorted(random.sample(range(size), 2))

    # Create child with None values
    child = [None] * size

    # Copy segment from parent1
    child[pt1:pt2+1] = solution1[pt1:pt2+1]

    # Fill the rest from parent2
    for i in range(size):
        if i < pt1 or i > pt2:
            gene = solution2[i]
            while gene in child:
                gene = solution2[solution1.index(gene)]  # Resolve conflict
            child[i] = gene

    return solution_saver(df, child)


def swap_mutation(individual, mutation_prob, df=file):
    i = 0
    j = 0
    individual = info(individual)
    status = False
    if random.random() < mutation_prob:
        while i == j:
            i = random.randint(0, len(individual) - 1)
            j = random.randint(0, len(individual) - 1)
        individual[i], individual[j] = individual[j], individual[i]
        status = True
    return solution_saver(df, individual), status



def epoch_create(epoch,mutation_rate, tournament_size):
    list_solutions = []
    list_fitness = []
    list_method = []
    while len(list_solutions) < len(epoch):
        # Selection
        parrent1 = tournament_selection(epoch,tournament_size)
        parrent2 = tournament_selection(epoch,tournament_size)
        # Crossover
        child = pmx_crossover(parrent1, parrent2, 0.9, file)
        # Mutation (W.I.P)
        mutated, status = swap_mutation(child,mutation_rate, file)
        # add that new child to epochn
        list_solutions.append(mutated)
        list_fitness.append(check_fitness(mutated))
        if status:
            list_method.append('M')
        else:
            list_method.append('C')
    new_epoch = {'solution': list_solutions, 'fitness': list_fitness,
                 'method': list_method}
    return pd.DataFrame(new_epoch)


def run_epochs(epoch0, num_epochs):
    row = 0
    population = epoch0
    best_solution = None
    best_score = float('inf')
    best_fitness_list = []
    prev_score = None
    epochs = []
    
    base_rate=0.2
    max_rate=0.2
    mutation_rate = base_rate
    no_improvement_count = 0
    tournament_size = 5

    # Set up live plots
    plt.ion()
    fig, (ax_fitness, ax_path) = plt.subplots(1, 2, figsize=(12, 6))
    ax_fitness.set_xlabel('Epoch')
    ax_fitness.set_ylabel('Best Fitness')
    ax_fitness.set_title('Best Fitness per Epoch')
    line_fitness, = ax_fitness.plot([], [], 'b-', label='Best Fitness')
    ax_fitness.legend()

    ax_path.set_title('Best Path in Current Epoch')
    path_line, = ax_path.plot([], [], 'r-', label='Path')
    scatter_points = ax_path.scatter([], [], c='blue', label='Cities')
    ax_path.legend()

    plt.show()

    for epoch in range(num_epochs):
        row += 1
        population = epoch_create(population,mutation_rate, tournament_size)
        epoch_best_score = min(population['fitness'])
        
        # if prev_score:
        #     if prev_score > epoch_best_score:
        #         no_improvement_count = 0
        #         mutation_rate = base_rate
        #         tournament_size = 3
        #         print('mutation back to base')
        #     elif prev_score <= epoch_best_score:
        #         no_improvement_count += 1
        #         if mutation_rate <= max_rate and no_improvement_count >= 5:
        #             mutation_rate += 0.1
        #             print('mutation increased')
        #         if tournament_size > 2 and no_improvement_count >= 9:
        #             tournament_size -= 1
        
        if epoch_best_score < best_score:
            best_score = epoch_best_score
            best_solution = population[population['fitness'] == min(population['fitness'])]

        # if no_improvement_count >= 25 and row > (num_epochs/100*50):
        #     print("stopping early due to no improvement.")
        #     break

        print(f"Epoch {epoch+1}: Best Score = {epoch_best_score}, Mutation Rate = {mutation_rate}, stable rounds: {no_improvement_count}, tournament size: {tournament_size}")
        population_info(population)
        prev_score = epoch_best_score
        print()



        # Update plots
        best_fitness_list.append(epoch_best_score)
        epochs.append(epoch + 1)
        line_fitness.set_xdata(epochs)
        line_fitness.set_ydata(best_fitness_list)
        ax_fitness.relim()
        ax_fitness.autoscale_view()

        x_coords = best_solution.iloc[0].iloc[0]['x'].to_list()
        x_coords.append(x_coords[0])
        y_coords = best_solution.iloc[0].iloc[0]['y'].to_list()
        y_coords.append(y_coords[0])
        path_line.set_data(x_coords, y_coords)
        scatter_points.set_offsets(best_solution.iloc[0].iloc[0][['x', 'y']].values)
        ax_path.relim()
        ax_path.autoscale_view()
        ax_path.set_title(f'Best Path in Epoch {epoch + 1}')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    return best_solution


file = pd.DataFrame(parser_tsp(file))

epoch0 = initial_population(file, 125)
population_info(epoch0)


# random_solutions = initial_population(file, 1000, 0)
# population_info(random_solutions)
# greedy_solutions = initial_population(file, 5, 1)
# population_info(greedy_solutions)

parrent1 = tournament_selection(epoch0,3)
parrent2 = tournament_selection(epoch0,3)
crossovered = pmx_crossover(parrent1, parrent2,0.9,file)
mutated, status = swap_mutation(crossovered,0.1,file)
# epoch1 = epoch_create(epoch0)

best = run_epochs(epoch0, 50)




# =============================================================================
# Plot section
# =============================================================================
#%%
import time

def test_hyperparameters(pop_size_list, mutation_rate_list, tournament_size_list, greedy_fraction_list, num_epochs=100):
    results = []
    
    for pop_size in pop_size_list:
        for mutation_rate in mutation_rate_list:
            for tournament_size in tournament_size_list:
                for greedy_fraction in greedy_fraction_list:
                    print(f"Testing: Pop={pop_size}, Mut={mutation_rate}, Tour={tournament_size}, Greedy={greedy_fraction}")
                    
                    # Initialize population
                    epoch0 = initial_population(file, pop_size, greedy_fraction)
                    
                    start_time = time.time()
                    best_solution = run_epochs(epoch0, num_epochs)
                    exec_time = time.time() - start_time
                    
                    best_fitness = min(epoch0['fitness'])
                    median_fitness = epoch0['fitness'].median()
                    
                    results.append({
                        'pop_size': pop_size,
                        'mutation_rate': mutation_rate,
                        'tournament_size': tournament_size,
                        'greedy_fraction': greedy_fraction,
                        'best_fitness': best_fitness,
                        'median_fitness': median_fitness,
                        'execution_time': exec_time
                    })
    
    return pd.DataFrame(results)

# Define hyperparameter ranges
test_results = test_hyperparameters(
    pop_size_list=[100, 150, 200],
    mutation_rate_list=[0.02, 0.05, 0.1],
    tournament_size_list=[3, 5, 7],
    greedy_fraction_list=[0.05, 0.1, 0.2],
    num_epochs=50
)

# Save results
test_results.to_csv("hyperparam_results.csv", index=False)
print(test_results)

#%%

# =============================================================================
# # sync testtt.
# =============================================================================
# print(info(random_df))


# print(find_distance(x1=(kroA100['x'][random.randint(0, len(kroA100)-1)]),
#                     y1= (kroA100['y'][random.randint(0, len(kroA100)-1)]),
#                     x2= (kroA100['x'][random.randint(0, len(kroA100)-1)]),
#                     y2= (kroA100['y'][random.randint(0, len(kroA100)-1)])))
