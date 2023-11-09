import pandas as pd
import random
import math
import turtle

cities = []
population = []

class Chromosome:
    def __init__(self, order, dist, fitness, norm_fitness):
        self.order = order
        self.dist = dist
        self.fitness = fitness
        self.norm_fitness = norm_fitness

    def duplicate(self):
        return Chromosome(self.order, self.dist, self.fitness, self.norm_fitness)

def main():
    global cities, population, df_5k, best_chromo
    city_num = 100
    population_size = 50
    generations = 10000
    mutation_rate = 0.2
    best_chromo = Chromosome([0], float('inf'), float('inf'), float('inf'))

    df = pd.read_csv("cities.csv")
    df_5k = df.iloc[:city_num]
    population = init_population(population_size, city_num)

#--------------MAIN-LOOP---------------#

    for i in range(0, generations):
        for j in range(0, len(population)):
            fitness_func(population[j])
        population.sort(key = lambda x: x.dist)
        print(i)

        if (population[0].dist < best_chromo.dist):
            best_chromo = population[0].duplicate()
            print("\n", best_chromo.dist, "\n")

        generate_norm_fitness()

        create_next_generation()
        mutate_generation(mutation_rate)

#--------------FUNCTIONS--------------#

def init_population(population_size, city_num):
    population.clear #Ensure population is empty
    for i in range(0, population_size):
        order = list(range(0, city_num))
        random.shuffle(order)
        chromo = Chromosome(order, float('inf'), float('inf'), 0)
        population.append(chromo)
    return population

def fitness_func(chromo):
    cumulative_dist = euclid_dist(chromo.order[len(chromo.order)-1], chromo.order[0])

    for i in range(0, len(chromo.order) -1):
        cumulative_dist = cumulative_dist + euclid_dist(chromo.order[i], chromo.order[i+1])
    chromo.dist = cumulative_dist
    chromo.fitness = (1 / cumulative_dist)**3

def generate_norm_fitness():
    total = 0
    for i in range(0, len(population)):
        total = total + population[i].fitness
    for i in range(0, len(population)):
        population[i].norm_fitness = population[i].fitness / total

        
def create_next_generation():
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
        order = population[i].order
        if (random.uniform(0, 1) < mutation_rate):
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

def euclid_dist_sqr(a, b):
    A = get_coords(a)
    B = get_coords(b)
    return (A[0] - B[0])**2 + (A[1] - B[1])**2

main()
print("Best Score: ", best_chromo.dist)