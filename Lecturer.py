import turtle
import random
import math

number_cities = 100
tabu_list = []

def main():
    random.seed(42)
    citylocations = []

    for i in range(0,number_cities):
        citylocations.append(
            (random.randint(-299, 299),
            random.randint(-299, 299)))

    screen = turtle.Screen()
    screen.setup(600,600)

    tabu_iteration(citylocations)

    screen.exitonclick()

def tabu_iteration(citylocations):
    candidates = create_candidates(citylocations)
    #best_score = 100000000
    #best_candidate = number_cities + 1

    scored_candidates = []

    scored_candidates = sorted(candidates, key=objective_function)


    for scored_candidate in scored_candidates:
        if not tabu_check(scored_candidate):
            usable_candidate = scored_candidate
            break

    drawpath(usable_candidate)
    #for i in range(0, number_cities):
        #scored_candidates.append((objective_function(candidates[i]), candidates[i]))
        #score = objective_function(candidates[i])
        #if score < best_score:
        #    best_score = score
        #    best_candidate = i

    drawpath(citylocations)

def tabu_check(candidate):
    global tabu_list

    match = False

    for tabu_candidate in tabu_list:
        if tabu_candidate == candidate:
           match = True 
           break

    return match


def objective_function(candidate):
    sum = 0

    for i in range(0, number_cities):
        if i == number_cities - 1:
            sum = sum + euclidean_distance(candidate[-1], candidate[0])
        else:
            sum = sum + euclidean_distance(candidate[i], candidate[i+1])

    return sum

def euclidean_distance(pointa, pointb):
    return math.sqrt(
        math.pow(pointa[0] - pointb[0], 2) + 
        math.pow(pointa[1] - pointb[1], 2))

def create_candidates(citylocations):
    candidates = []

    for i in range(0, number_cities):

        candidate = []

        if i == number_cities - 1:
            candidate.append(citylocations[-1])
            candidate.extend(citylocations[1:-1])
            candidate.append(citylocations[0])
        else:
            for j in range(0, number_cities):
                if not j == i and not j == i + 1:
                    candidate.append(citylocations[j])
                elif j == i and not i == number_cities - 1:
                    candidate.append(citylocations[i+1])
                    candidate.append(citylocations[i])

            if i == number_cities - 1:
                candidate.append(citylocations[0])

        candidates.append(candidate)

    return candidates


def drawpath(cities):
    turtle.clear()
    turtle.penup()

    for city in cities:
        turtle.goto(city)
        turtle.pendown()

    turtle.goto(cities[0])


if __name__ == '__main__':
    main()