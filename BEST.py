import pandas as pd
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt

class Chromosome:
    def __init__(self, order, dist, fitness):
        self.order = order
        self.dist = dist
        self.fitness = fitness

    def duplicate(self):
        return Chromosome(self.order.copy(), self.dist, self.fitness)

def main():
    city_num = 5000
    pop_size = 400
    generations = 1000
    tournament_size = 15
    mutation_rate = .1
    mutation_reps = 2
    best_chromo = Chromosome(np.arange(city_num), 20000000, float('inf'))

    matrix = np.zeros((city_num, city_num))

    target = 10500000

    df = pd.read_csv("cities.csv")
    df_5k = df.iloc[:city_num]
    population = init_population(pop_size, city_num)

    start_time = time.time()

    i = -1
    while (best_chromo.dist >= 2500000): 
        i+=1
    #for i in range(generations):
        start_fit = time.time()


        for chromo in population:
            fitness_func(chromo, df_5k, matrix)

        best_chromo_gen = max(population, key=lambda x: x.fitness)

        if best_chromo_gen.dist < best_chromo.dist:
            print("######################################################", math.floor(best_chromo.dist - best_chromo_gen.dist))
            best_chromo = best_chromo_gen.duplicate()

            if best_chromo.dist < target:
                time_convert(time.time() - start_time, str(target))
                target -= 100000

        print(i, " : ", math.floor(best_chromo.dist))
        #time_convert(time.time() - start_fit, "")

        population = crossover_generation(population, tournament_size)
        mutate_generation(mutation_rate, population, mutation_reps)

        #time_convert(time.time() - start_fit, "Generation")


    end_time = time.time()
    print(time_convert(end_time - start_time, "Total Time"), "Best Score: ", best_chromo.dist)
    plot_path(best_chromo, df_5k, "Best Found Path")

    #FUNCTIONS

def plot_path(chromo, df_5k, title):
    order = chromo.order
    coords = np.array([get_coords(index, df_5k) for index in order])
    plt.figure()
    plt.plot(coords[:, 0], coords[:, 1], marker='o', linestyle='-')
    title = title, chromo.dist
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

def crossover_generation(population, tournament_size):
    new_population = np.empty(len(population), dtype=Chromosome)

    for i in range(len(population)):
        parent_a = select_parent(population, tournament_size)
        parent_b = select_parent(population, tournament_size)
        new_population[i] = crossover(parent_a, parent_b)
    return new_population

def select_parent(population, tournament_size):
    tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
    tournament = [population[idx] for idx in tournament_indices]
    parent = max(tournament, key=lambda x: x.fitness)
    return parent.duplicate()

def mutate_generation(mutation_rate, population, reps):
    for chromo in population:
        if (random.uniform(0, 1) < mutation_rate):
            for j in range(0, reps):
                x, y = np.random.choice(len(chromo.order), size=2, replace=False)
                #chromo.order[x], chromo.order[y] = chromo.order[y], chromo.order[x]
                chromo.order = np.insert(np.delete(chromo.order, x), y, chromo.order[x])



def crossover(parent_a, parent_b):
    order_a = parent_a.order
    order_b = parent_b.order

    crossover_point = random.randint(0, len(order_a)-1)
    cities_to_add = np.setdiff1d(order_b, order_a[:crossover_point], assume_unique=True)
    new_order = np.concatenate((order_a[:crossover_point], cities_to_add))

    return Chromosome(new_order, float('inf'), float('inf'))

def init_population(population_size, city_num):
    population = np.empty(population_size, dtype=Chromosome)
    for i in range(0, population_size):
        order = np.arange(0, city_num)
        random.shuffle(order)
        chromo = Chromosome(order, float('inf'), float('inf'))
        population[i] = chromo
    return population

def fitness_func(chromo, df_5k, matrix):
    cumulative_dist = euclid_dist(chromo.order[len(chromo.order)-1], chromo.order[0], df_5k, matrix)
    for i in range(0, len(chromo.order) -1):
        cumulative_dist += euclid_dist(chromo.order[i], chromo.order[i+1], df_5k, matrix)
    chromo.dist = cumulative_dist
    chromo.fitness = (1 / cumulative_dist)

def get_coords(index, df_5k):
    return ([df_5k.loc[index, "X"], df_5k.loc[index, "Y"]])

def euclid_dist(a, b, df_5k, matrix):
    if (matrix[a][b] == 0 and matrix[b][a] == 0):

        A = get_coords(a, df_5k)
        B = get_coords(b, df_5k)
        dist = math.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
        matrix[a][b] = dist
        matrix[b][a] = dist
        return dist
    else:
        return matrix[a][b]

def time_convert(sec, message):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print(message, "= {0}:{1}:{2}".format(int(hours),int(mins),sec))

if __name__ == "__main__":
    main()