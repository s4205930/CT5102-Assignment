import pandas as pd
import random
import math
import turtle
import time

#Distance to beat: 10987144.136907717

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
    global cities, population, df_5k, best_chromo, best_from_gen
    city_num = 500
    population_size = 5
    generations = 1000
    mutation_rate = 0.2
    best_chromo = Chromosome([0], float('inf'), float('inf'), float('inf'))
    #best_from_gen = Chromosome([0], float('inf'), float('inf'), float('inf'))
    #10904183.941972187 - 100:50 for 5000
    #10869285.034202613 - 50:100 for 5000

    df = pd.read_csv("cities.csv")
    df_5k = df.iloc[:city_num]
    population = init_population(population_size, city_num)

    #TURTLE
    #screen = turtle.Screen()
    #screen.setup(800, 800)
    

#--------------MAIN-LOOP---------------#
    start_time = time.time()

    for i in range(0, generations):
        for j in range(0, len(population)):
            fitness_func(population[j])
        population.sort(key = lambda x: x.dist)
        print(population[0].dist, " : ", i)

        if (population[0].dist < best_chromo.dist):
            best_chromo = population[0].duplicate()
            print("\n", best_chromo.dist, "\n")

        generate_norm_fitness()

        create_next_generation()
        mutate_generation(mutation_rate)
    
    end_time = time.time()
    print(time_convert(end_time - start_time))
    

#--------------FUNCTIONS--------------#

def draw_path(chromo):
    order = chromo.order
    for i in range(0, len(order)):
        pass

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Elapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

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
        cumulative_dist += euclid_dist(chromo.order[i], chromo.order[i+1])
    chromo.dist = cumulative_dist
    chromo.fitness = (1 / cumulative_dist)**10

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
#screen.exitonclick()