import numpy as np
import random
from math import exp

f = open('output.txt', 'a+')

def environment_setup(size_of_grid,wall_coordinates,negative_reward_coordinates,positive_reward_coordinates):
    environment = np.zeros(shape = size_of_grid)
    #999 implies Wall
    for i in wall_coordinates:
        environment[i] = 999

    for i in negative_reward_coordinates:
        environment[i] = -1

    for i in positive_reward_coordinates:
        environment[i] = 1    
    return environment

def positional_values_for_cells(environment):
    i = 0
    environment_with_positional_values = np.zeros(shape = environment.shape, dtype = int)
    for row in range(environment.shape[0]):
        for col in range(environment.shape[1]):
            environment_with_positional_values[row][col] = i
            i +=1
    return environment_with_positional_values

def possible_actions_at_given_state(given_state, environment):
    #Actions 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
    
    x = given_state[0]
    y = given_state[1]
    
    size_of_grid = environment.shape
    size_of_grid_x = size_of_grid[0]
    size_of_grid_y = size_of_grid[1]
    max_x = max(range(size_of_grid_x + 1))
    min_x = min(range(size_of_grid_x + 1))
    max_y = max(range(size_of_grid_y + 1))
    min_y = min(range(size_of_grid_y + 1))
    
    possible_actions = []
    if x - 1 >= min_x and environment[x-1][y]!=999:
        possible_actions.append(0) 
    if x + 1 < max_x and environment[x+1][y]!=999:
        possible_actions.append(1)
    if y - 1 >= min_y and environment[x][y-1]!=999:
        possible_actions.append(2)
    if y + 1 < max_y and environment[x][y+1]!=999:
        possible_actions.append(3)  
    
    return possible_actions

def next_state_coordinates(current_state, selected_action):
    
    x = current_state[0]
    y = current_state[1]
    next_state = (0,0)
    
    if not isinstance(selected_action, int):
        selected_action = selected_action[0]
    
    if selected_action == 0:
        next_state = (x-1, y)
    elif selected_action == 1:
        next_state = (x+1, y)
    elif selected_action == 2:
        next_state = (x, y-1)
    else:
        next_state = (x, y+1)
 
    return next_state

def rewards_current_state(current_state, environment):
    reward = environment[current_state]
    return reward

def bellman_equation(state_index, action, reward, q_max, q_table):
    alpha = 0.01
    beta = 0.9
    q_table[state_index][action] = q_table[state_index][action] + alpha * (reward + (beta * q_max) - q_table[state_index][action])
    return q_table[state_index][action]

def print_results(number_of_steps, q, epsilon, temperature_values):
    print(file = f)
    temperature_values_unique = np.unique(temperature_values)
    if epsilon is None:
        print("The temperature T value is started at T = 10. After every 10 steps the temperature is reduced by 0.05" , file = f)
        print(file = f)
        print("Unique temperature values used", file = f)
        print(-np.sort(-temperature_values_unique), file = f)
        print(file = f)
        print("Total unique temperature values = %d" %(len(temperature_values_unique)), file = f)
        print(file = f)
    else:
        print("Epsilon = %f" %(epsilon), file = f)

    print("Number of iterations: %d" %(number_of_steps), file = f)
    print("Q-table", file = f)
    print(q, file = f)

def q_learning_algorithm_greedy(environment, epsilon, environment_with_positional_values):
    number_of_steps = 0
    current_state = (0,0)
    reward = rewards_current_state(current_state, environment)
    q = np.zeros(shape = (100,4))
    
    while reward != 1.0:
        reward = rewards_current_state(current_state, environment)
        current_state_possible_actions = possible_actions_at_given_state(current_state, environment)
        current_state_index = environment_with_positional_values[current_state]
        next_state = (0,0)
        selected_action = []
        q_max_exploit = 0
        q_max = 0
        
        if len(current_state_possible_actions) > 0:
            #Exploit
            if random.uniform(0, 1) <= epsilon:
                for action in current_state_possible_actions:
                    if q_max_exploit <= q[current_state_index][action]:
                        q_max_exploit = q[current_state_index][action]
                        selected_action = action
            #Explore
            else:
                selected_action = random.sample(current_state_possible_actions, 1)

            next_state = next_state_coordinates(current_state, selected_action) 
            next_state_index = environment_with_positional_values[next_state]
            next_state_possible_actions = possible_actions_at_given_state(next_state, environment)
           
            for action in next_state_possible_actions:
                if q_max <= q[next_state_index][action]:
                    q_max = q[next_state_index][action]
                    
            q[current_state_index][action] = bellman_equation(current_state_index, action, reward, q_max, q)
            number_of_steps += 1
            current_state = next_state
            
    return number_of_steps, q

def boltzmann_exponential_calculation(q_value, temperature):
    return exp(q_value/temperature)

def boltzmann_probability(q_table,temperature,number_of_steps,current_state_index,current_state_possible_actions):
    if number_of_steps>0 and number_of_steps % 10 == 0 and temperature!=0:
        temperature = temperature - 0.05
        
    probabilities_for_actions = []
     
    all_denominator_values = []
    
    for action in current_state_possible_actions:
        q_value = q_table[current_state_index][action]
        denominator = boltzmann_exponential_calculation(q_value,temperature)
        all_denominator_values.append(denominator)
    
    sum_denominator = np.sum(all_denominator_values)

    for action in current_state_possible_actions:
        q_value = q_table[current_state_index][action]
        numerator = boltzmann_exponential_calculation(q_value,temperature)
        if sum_denominator != 0:
            probability = numerator/sum_denominator
        else:
            probability = 0.0
        probabilities_for_actions.append(probability)
    
    return probabilities_for_actions, temperature

def q_learning_algorithm_boltzman(environment, environment_with_positional_values):
    number_of_steps = 0
    current_state = (0,0)
    reward = rewards_current_state(current_state, environment)
    q = np.zeros(shape = (100,4))
    temperature = 10
    all_temperature_values = []
    
    while reward != 1.0:
        reward = rewards_current_state(current_state, environment)
        current_state_possible_actions = possible_actions_at_given_state(current_state, environment)
        current_state_index = environment_with_positional_values[current_state]
        next_state = (0,0)
        selected_action = []
        q_max = 0
        
        if len(current_state_possible_actions) > 0:
            
            probabilities,temperature = boltzmann_probability(q, temperature, number_of_steps, current_state_index, current_state_possible_actions)
            maximum_prob = max(probabilities)
            minimum_prob = min(probabilities)
            
            if maximum_prob - minimum_prob <= 0.001:
                selected_action = random.sample(current_state_possible_actions, 1)
            else:
                selected_action = probabilities.index(maximum_prob)
                
            next_state = next_state_coordinates(current_state, selected_action)          
            next_state_index = environment_with_positional_values[next_state]
            next_state_possible_actions = possible_actions_at_given_state(next_state, environment)
           
            for action in next_state_possible_actions:
                if q_max <= q[next_state_index][action]:
                    q_max = q[next_state_index][action]
                    
            q[current_state_index][action] = bellman_equation(current_state_index, action, reward, q_max, q)
            number_of_steps += 1
            current_state = next_state
            all_temperature_values.append(temperature)
            
    return number_of_steps, q, all_temperature_values

def main():
    #Environment Setup
    size_of_grid = 10,10
    wall_coordinates = [(2,1),(2,2),(2,3),(2,4), (2,6), (2,7), (2,8), (3,4), (4,4), (5,4), (6,4), (7,4)]
    negative_reward_coordinates = [(3,3),(4,5),(4,6),(5,6),(5,8),(6,8),(7,3),(7,5),(7,6)]
    positive_reward_coordinates = [(5,5)]
    environment = environment_setup(size_of_grid,wall_coordinates,negative_reward_coordinates,positive_reward_coordinates)
    environment_with_positional_values = positional_values_for_cells(environment)
    
    print("Gridworld Q-Learning", file = f)
    print(file = f)
    print("Gridworld Environment", file = f)
    print(environment, file = f)
    print(file = f)
    print("Size of the environment", file = f)
    print(environment.shape, file = f)
    
    epsilon_1 = 0.1
    number_of_steps_1, q_1 = q_learning_algorithm_greedy(environment,epsilon_1,environment_with_positional_values)
    epsilon_2 = 0.2
    number_of_steps_2, q_2 = q_learning_algorithm_greedy(environment,epsilon_2,environment_with_positional_values)
    epsilon_3 = 0.3
    number_of_steps_3, q_3 = q_learning_algorithm_greedy(environment,epsilon_3,environment_with_positional_values)
    
    print(file = f)
    print("Epsilon Greedy Exploration Policy", file = f)
    print_results(number_of_steps_1, q_1, epsilon_1, None)
    print_results(number_of_steps_2, q_2, epsilon_2, None)
    print_results(number_of_steps_3, q_3, epsilon_3, None)
    
    number_of_steps_b, q_b, temperature_values = q_learning_algorithm_boltzman(environment, environment_with_positional_values)
    
    print(file = f)
    print("Boltzman Exploration Policy", file = f)
    print_results(number_of_steps_b, q_b, None, temperature_values)

main()