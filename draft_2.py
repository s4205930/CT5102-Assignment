import pandas as pd
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt


class Chromosome:
    def __init__(self, order, dist, fitness, norm_fitness):
        self.order = order
        self.dist = dist
        self.fitness = fitness
        self.norm_fitness = norm_fitness

    def duplicate(self):
        return Chromosome(self.order, self.dist, self.fitness, self.norm_fitness)

def main():
    city_num = 5000
    pop_size = 25
    generations = 1000
    mutation_rate = 0.2
    best_chromo = Chromosome([0], 20000000, float('inf'), float('inf'))

    df = pd.read_csv("cities.csv")
    df_5k = df.iloc[:city_num]
    population = init_population(pop_size, city_num)

#--------------MAIN-LOOP---------------#
    start_time = time.time()

    for i in range(0, generations):
    #i = 0
    #while(best_chromo.dist > 9000000):
        start_fit = time.time()
        for j in range(0, len(population)):
            fitness_func(population[j], df_5k)
        time_convert(time.time()-start_fit, "Fitness")
        sorted_indices_desc = np.argsort([chromo.dist for chromo in population])[::1]
        population = population[sorted_indices_desc]
        print(math.floor(population[0].dist), " : ", i, " : ", math.floor(best_chromo.dist))

        if (population[0].dist < best_chromo.dist):
            best_chromo = population[0].duplicate()
            print("\n")

        generate_norm_fitness(population)

        create_next_generation_roulette(population)
        mutate_generation(mutation_rate, population)
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
    population = np.empty(population_size, dtype=Chromosome)
    for i in range(0, population_size):
        order = np.arange(0, city_num)
        random.shuffle(order)
        chromo = Chromosome(order, float('inf'), float('inf'), 0)
        population[i] = chromo
    return population

def fitness_func(chromo, df_5k):
    cumulative_dist = euclid_dist(chromo.order[len(chromo.order)-1], chromo.order[0], df_5k)
    for i in range(0, len(chromo.order) -1):
        cumulative_dist += euclid_dist(chromo.order[i], chromo.order[i+1], df_5k)
    chromo.dist = cumulative_dist
    chromo.fitness = (1 / cumulative_dist)

def generate_norm_fitness(population):
    total = 0
    for i in range(0, len(population)):
        total = total + population[i].fitness
    for i in range(0, len(population)):
        population[i].norm_fitness = population[i].fitness / total

        
def create_next_generation_roulette(population):#tournament & numpy fitness
    new_population = np.empty(len(population), dtype=Chromosome)

    for i in range(0, len(population)):
        chromo_select = random.uniform(0, 1)
        index = -1

        while (chromo_select > 0):
            index += 1
            chromo_select -= population[index].norm_fitness

        new_population[i] = (population[index])

    population = new_population

def mutate_generation(mutation_rate, population):
    for i in range(0, len(population)):
        if (random.uniform(0, 1) < mutation_rate):
            reps = random.randrange(math.floor(5))
            for j in range(0, reps):
                x = random.randrange(len(population[i].order)-1)

                temp = population[i].order[x]
                population[i].order[x] = population[i].order[x+1]
                population[i].order[x+1] = temp


def get_coords(index, df_5k):
    return ([df_5k.loc[index, "X"], df_5k.loc[index, "Y"]])

def euclid_dist(a, b, df_5k):
    A = get_coords(a, df_5k)
    B = get_coords(b, df_5k)
    return math.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)


main()