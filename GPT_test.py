import pandas as pd
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from numba import jit, cuda

class Chromosome:
    def __init__(self, order, dist, fitness, norm_fitness, coordinates=None):
        self.order = order
        self.dist = dist
        self.fitness = fitness
        self.norm_fitness = norm_fitness
        self.coordinates = coordinates

    def duplicate(self):
        return Chromosome(self.order.copy(), self.dist, self.fitness, self.norm_fitness, self.coordinates.copy())

def main():
    city_num = 5000
    pop_size = 20
    generations = 10
    mutation_rate = 0.2
    best_chromo = Chromosome([0], 20000000, float('inf'), float('inf'))

    df = pd.read_csv("cities.csv")
    df_5k = df.iloc[:city_num]
    population = init_population(pop_size, city_num, df_5k)

    # Use a set to store visited orders for duplicate checking
    visited_orders = set(tuple(chromo.order) for chromo in population)

    # Set up the Matplotlib figure for real-time updates
    plt.ion()
    fig, ax = plt.subplots()

    start_time = time.time()

    for i in range(generations):
        start_fit = time.time()

        for chromo in population:
            fitness_func(chromo, df_5k)

        time_convert(time.time() - start_fit, "Fitness")

        sorted_indices = np.argsort([chromo.dist for chromo in population])[::1]
        population = population[sorted_indices]

        if population[0].dist < best_chromo.dist:
            best_chromo = population[0].duplicate()
            ax.clear()
            plot_path(best_chromo, df_5k, ax, f"Generation {i + 1} - Best Path")
            plt.pause(0.01)  # Pause to allow time for the plot to update

        print(math.floor(population[0].dist), " : ", i, " : ", math.floor(best_chromo.dist))

        generate_norm_fitness(population)

        create_next_generation_tournament(population, 5)
        mutate_generation(mutation_rate, population)

    # Turn off interactive mode and show the final plot
    plt.ioff()
    plt.show()

    end_time = time.time()
    print(time_convert(end_time - start_time, "Total Time"))
    plot_path(best_chromo, df_5k, "Best Found Path")

def plot_path(chromo, df_5k, ax, title):
    order = chromo.order
    coords = np.array([get_coords(index, df_5k) for index in order])
    ax.plot(coords[:, 0], coords[:, 1], marker='o', linestyle='-')
    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    plt.pause(0.01)  # Pause to allow time for the plot to update

def create_next_generation_tournament(population, tournament_size):
    new_population = np.empty(len(population), dtype=Chromosome)

    for i in range(len(population)):
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament = [population[idx] for idx in tournament_indices]
        winner = max(tournament, key=lambda x: x.norm_fitness)
        new_population[i] = winner.duplicate()

    return new_population

def mutate_generation(mutation_rate, population):
    for chromo in population:
        if random.uniform(0, 1) < mutation_rate:
            reps = random.randrange(5)
            for j in range(0, reps):
                x = random.randrange(len(chromo.order) - 1)

                temp = chromo.order[x]
                chromo.order[x] = chromo.order[x + 1]
                chromo.order[x + 1] = temp

def init_population(population_size, city_num, df_5k):
    population = np.empty(population_size, dtype=Chromosome)
    for i in range(0, population_size):
        order = np.arange(0, city_num)
        random.shuffle(order)
        chromo = Chromosome(order, float('inf'), float('inf'), 0, coordinates=[get_coords(i, df_5k) for i in order])
        population[i] = chromo
    return population

def fitness_func(chromo, df_5k):
    order = chromo.order
    coordinates = np.array([get_coords(i, df_5k) for i in order])
    cumulative_dist = np.sum([euclid_dist(order[i], order[i + 1], df_5k) for i in range(len(order) - 1)])
    cumulative_dist += euclid_dist(order[-1], order[0], df_5k)
    chromo.dist = cumulative_dist
    chromo.fitness = 1 / cumulative_dist

def generate_norm_fitness(population):
    total = 0
    for i in range(0, len(population)):
        total = total + population[i].fitness
    for i in range(0, len(population)):
        population[i].norm_fitness = population[i].fitness / total

def get_coords(index, df_5k):
    return [df_5k.loc[index, "X"], df_5k.loc[index, "Y"]]

def euclid_dist(a, b, df_5k):
    A = get_coords(a, df_5k)
    B = get_coords(b, df_5k)
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

def time_convert(sec, message):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print(message, "= {0}:{1}:{2}".format(int(hours), int(mins), sec))

main()
