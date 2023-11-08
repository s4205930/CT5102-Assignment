import pandas as pd
import random
import math

cities = []
population = []

class Chromosome:
    def __init__(self, order, fitness):
        self.order = order
        self.fitness = fitness

def main():
    global cities, population
    city_num = 50
    population_size = 5

    df = pd.read_csv("cities.csv")
    df_5k = df.iloc[:city_num]
    print(df_5k.head(10))
    print(df_5k.shape)
    population = init_population(population_size, city_num)

def init_population(population_size, city_num):
    population.clear #Ensure population is empty
    for i in range(0, population_size):
        chromo = Chromosome(list(range(0, city_num)), float('inf'))
        population.append(chromo)
    return population


main()