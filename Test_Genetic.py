import pandas as pd
import random
import math

cities = []
population = []

class Chromosome:
    def __init__(self, order, fitness, norm_fitness):
        self.order = order
        self.fitness = fitness
        self.norm_fitness = norm_fitness

def main():
    global cities, population, df_5k
    city_num = 5000
    population_size = 10
    generations = 100

    df = pd.read_csv("cities.csv")
    df_5k = df.iloc[:city_num]
    population = init_population(population_size, city_num)

    for i in range(0, generations):
        for j in range(0, len(population)):
            fitness_func(population[j])
        population.sort(key = lambda x: x.fitness)
        print(population[0].fitness)
        generate_norm_fitness()
        create_next_generation()



def init_population(population_size, city_num):
    population.clear #Ensure population is empty
    for i in range(0, population_size):
        order = list(range(0, city_num))
        random.shuffle(order)
        chromo = Chromosome(order, float('inf'), 0)
        population.append(chromo)
    return population

def fitness_func(chromo):
    cumulative_dist = euclid_dist(chromo.order[len(chromo.order)-1], chromo.order[0])

    for i in range(0, len(chromo.order) -1):
        cumulative_dist = cumulative_dist + euclid_dist(chromo.order[i], chromo.order[i+1])
    chromo.fitness = cumulative_dist

def generate_norm_fitness():
    total = 0
    for i in range(0, len(population)):
        total = total + population[i].fitness
    for i in range(0, len(population)):
        population[i].norm_fitness = population[i].fitness / total
        #print(population[i].norm_fitness)

        
def create_next_generation():
    global population
    new_population = []

    for i in range(0, len(population)):
        chromo_select = random.uniform(0, 1)
        index = -1

        while (chromo_select > 0):
            index += 1
            if ((chromo_select - population[index].norm_fitness) > 0):
                break
        new_population.append(population[index])

    population = new_population


def mutate_next_generation():
    pass

def get_coords(index):
    return ([df_5k.loc[index, "X"], df_5k.loc[index, "Y"]])

def euclid_dist(a, b):
    A = get_coords(a)
    B = get_coords(b)
    return math.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)

main()