import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import math

coefficient_lower_bound = -50
coefficient_upper_bound = 50
population_size = 50


def draw_plot(x, y):
    np.random.seed(19680801)
    N = len(x)
    colors = np.random.rand(N)
    area = (30 * np.random.rand(N)) ** 2  # 0 to 15 point radii
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.show()


def draw_quadratic_graph(coeffs):
    x = np.linspace(-10, 10, 1000)
    N = len(x)
    colors = np.random.rand(N)
    y = (coeffs[0])*(x ** 2) + coeffs[1] * x + coeffs[2]
    plt.scatter(x, y, c=colors, alpha=0.5)
    plt.show()


def initialize_population():
    initial_population = []
    for i in range(population_size):
        initial_population += [np.random.uniform(low=coefficient_lower_bound,
                                                 high=coefficient_upper_bound,
                                                 size=(3,))]
    return initial_population


def crossover_util(parent1, parent2):
    return (parent1 + parent2) / 2


def crossover(population):
    for i in range(math.floor(population_size / 2)):
        parent1 = population[randrange(population_size)]
        parent2 = population[randrange(population_size)]
        population += [crossover_util(parent1, parent2)]
    return population


def mutate(population):
    mutation_probability = 0.1
    for i in range(math.floor(population_size * mutation_probability)):
        child = population[randrange(population_size)]
        sign = (-1) ** (randrange(10000))
        mutated = child + sign * mutation_probability * coefficient_upper_bound
        population += [mutated]
    return population


def select(population, x, y):
    sorted_population = sorted(population, key=lambda coeff: fitness(x, y, coeff))
    return sorted_population[:population_size]


def fitness(x, y, coeff):
    X = np.array([x ** 2, x, 1])
    y_hat = np.dot(X, coeff)
    error = 0
    for i in range(len(y)):
        error += (y[i] - y_hat[i]) ** 2
    error = math.sqrt(error / len(y))
    return error


def average_fitness(x, y, population):
    avg = 0
    for p in population:
        avg += fitness(x, y, p)
    return avg / population_size


def main():
    x = np.genfromtxt('x_train.csv', delimiter=',')
    y = np.genfromtxt('y_train.csv', delimiter=',')
    population = initialize_population()
    # draw_plot(x,y)
    # print(population)
    num_of_iterations = 50
    for i in range(num_of_iterations):
        population = crossover(population)
        # print("after crossover: " + str(population))
        population = mutate(population)
        # print("after mutation" + str(population))
        population = select(population, x, y)
        # print("after selection" + str(population))
        print("best fitness of this gen is: " + str(fitness(x, y, population[0])))
        print("average fitness of this gen is: " + str(average_fitness(x, y, population)))
        print("best value of this gen is: " + str(population[0]))
        print("*"*100)
    draw_quadratic_graph(population[0])


main()