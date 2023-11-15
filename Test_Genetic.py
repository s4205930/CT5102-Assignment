import pandas as pd
import numpy as np
import random
import math
import time
#from numba import jit, cuda

#Distance to beat: 10987144.136907717

cities = np.array
population = np.array

class Chromosome:
    def __init__(self, order, dist, fitness, norm_fitness):
        self.order = order
        self.dist = dist
        self.fitness = fitness
        self.norm_fitness = norm_fitness

    def duplicate(self):
        return Chromosome(self.order, self.dist, self.fitness, self.norm_fitness)

def main():
    global cities, population, df_5k, best_chromo, look_up
    city_num = 5000
    population_size = 25
    generations = 5
    mutation_rate = 0.3
    look_up = np.zeros((city_num, city_num))
    best_chromo = Chromosome([0], 100000000, float('inf'), float('inf'))

    df = pd.read_csv("cities.csv")
    df_5k = df.iloc[:city_num]
    population = init_population(population_size, city_num)

    

#--------------MAIN-LOOP---------------#
    start_time = time.time()
    calculate_look_up(city_num)

    for i in range(0, generations):
    #i = 0
    #while(best_chromo.dist > 10000000):
        start_fit = time.time()
        best_gen_index = 0
        for j in range(0, len(population)):
            best_gen_index = fitness_func(population[j], best_gen_index)
        time_convert(time.time()-start_fit, "Fitness")
        #population.sort()
        print(math.floor(population[0].dist), " : ", i, " : ", math.floor(best_chromo.dist))

        if (population[0].dist < best_chromo.dist):
            best_chromo = population[0].duplicate()
            print("\n")

        generate_norm_fitness()

        create_next_generation_roulette()
        mutate_generation(mutation_rate)
        i += 1
    
    end_time = time.time()
    print(time_convert(end_time - start_time, "Total Time"))
    

#--------------FUNCTIONS--------------#

def time_convert(sec, message):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print(message, "= {0}:{1}:{2}".format(int(hours),int(mins),sec))

def init_population(population_size, city_num):
    global population
    population = np.empty(population_size, dtype=Chromosome)
    for i in range(0, population_size):
        order = np.arange(0, city_num)
        random.shuffle(order)
        chromo = Chromosome(order, float('inf'), float('inf'), 0)
        population[i] = chromo
    return population

def fitness_func_precalc(chromo):
    cumulative_dist = look_up[len(chromo.order) -1][chromo.order[0]]

    for i in range (0, len(chromo.order)-1):
        cumulative_dist += look_up[chromo.order[i]][chromo.order[i+1]]
    chromo.dist = cumulative_dist
    chromo.fitness = (1 / cumulative_dist)**4

def fitness_func(chromo, index):
    cumulative_dist = euclid_dist(chromo.order[len(chromo.order)-1], chromo.order[0])
    for i in range(0, len(chromo.order) -1):
        cumulative_dist += euclid_dist(chromo.order[i], chromo.order[i+1])
    chromo.dist = cumulative_dist
    chromo.fitness = (1 / cumulative_dist)

def generate_norm_fitness():
    total = 0
    for i in range(0, len(population)):
        total = total + population[i].fitness
    for i in range(0, len(population)):
        population[i].norm_fitness = population[i].fitness / total

        
def create_next_generation_roulette():#tournament & numpy fitness
    global population
    new_population = []

    for i in range(0, len(population)):
        chromo_select = random.uniform(0, 1)
        index = -1

        while (chromo_select > 0):
            index += 1
            chromo_select -= population[index].norm_fitness

        new_population.append(population[index])

    population = new_population

def mutate_generation(mutation_rate):
    for i in range(0, len(population)):
        if (random.uniform(0, 1) < mutation_rate):
            order = population[i].order
            reps = random.randrange(math.floor(5))
            for j in range(0, reps):

                a = random.randrange(len(order)-1)
                b = random.randrange(len(order)-1)

                temp = order[a]
                order[a] = order[b]
                order[b] = temp

def get_coords(index):
    return ([df_5k.loc[index, "X"], df_5k.loc[index, "Y"]])

def euclid_dist(a, b):
    A = get_coords(a)
    B = get_coords(b)
    return math.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)

def calculate_look_up(city_num):
    global look_up
    try:
        df = pd.read_csv('look_up.csv')
        look_up = df.to_numpy()
    except:
        for i in range(0, city_num):
            print(i)
            for j in range(0, city_num):
                look_up[i][j] = euclid_dist(i, j)
        df = pd.DataFrame(look_up)
        df.to_csv('look_up.csv', index = False)


main()
print("Best Score: ", best_chromo.dist)