# Imports necessary libraries
import random
import copy
import matplotlib.pyplot as plt

# Defines a class representing an individual in the genetic algorithm
class Individual:
    def __init__(self, N, MIN=-10, MAX=10):
        # Initialize an individual with a random gene and fitness value
        self.gene = [random.uniform(MIN, MAX) for _ in range(N)]
        self.fitness = 0

# Defines the test function for evaluating an individual's fitness(Code for the first equation)
def test_function_one(ind):
    utility = 0

    # Calculates fitness based on a specific formula
    for i in range(1, len(ind.gene)):
        utility = utility + (i * (((2 * (ind.gene[i] ** 2)) - ind.gene[i - 1]) ** 2))

    return (utility + ((ind.gene[0] - 1) ** 2))

# Creates an initial population of individuals
def create_initial_population(P, N, MIN=-10, MAX=10):
    return [Individual(N, MIN, MAX) for _ in range(P)]

# Performs tournament selection to choose parents for reproduction
def tournament_selection(population, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        winner = min(tournament, key=lambda ind: ind.fitness)
        selected_parents.append(copy.deepcopy(winner))
    return selected_parents

# Performs real-valued mutation on offspring
def real_valued_mutation(offspring, P, N, mutation_rate, MUTSTEP, MIN, MAX):
    for i in range(P):
        new_ind = Individual(N, MIN, MAX)
        new_ind.gene = []

        for j in range(N):
            gene = offspring[i].gene[j]
            mut_prob = random.random()

            if mut_prob < mutation_rate:
                alter = random.uniform(-MUTSTEP, MUTSTEP)
                gene = gene + alter

                # Ensures the gene is within the given range [MIN, MAX]
                gene = max(MIN, min(MAX, gene))

            new_ind.gene.append(gene)

        offspring[i] = copy.deepcopy(new_ind)

# Evaluates the fitness of the whole population
def evaluate_population(population, fitness_function):
    for ind in population:
        ind.fitness = fitness_function(ind)

# Performs one-point crossover on offspring
def one_point_crossover(offspring, P, N):
    toff1 = Individual(N)
    toff2 = Individual(N)
    temp = Individual(N)

    for i in range(0, P, 2):
        toff1 = copy.deepcopy(offspring[i])
        toff2 = copy.deepcopy(offspring[i + 1])
        temp = copy.deepcopy(offspring[i])
        crosspoint = random.randint(1, N)

        for j in range(crosspoint, N):
            toff1.gene[j] = toff2.gene[j]
            toff2.gene[j] = temp.gene[j]

        offspring[i] = copy.deepcopy(toff1)
        offspring[i + 1] = copy.deepcopy(toff2)

# Implements the main genetic algorithm
def genetic_algorithm(P, N, generations, tournament_size, crossover_rate, mutation_rate, MUTSTEP, MIN, MAX):
    population = create_initial_population(P, N, MIN, MAX)
    best_fitness_list = []
    mean_fitness_list = []

    for generation in range(generations):
        evaluate_population(population, test_function_one)

        best_fitness = max(ind.fitness for ind in population)
        mean_fitness = sum(ind.fitness for ind in population) / P

        best_fitness_list.append(best_fitness)
        mean_fitness_list.append(mean_fitness)

        offspring = tournament_selection(population, tournament_size)
        one_point_crossover(offspring, P, N)
        real_valued_mutation(offspring, P, N, mutation_rate, MUTSTEP, MIN, MAX)

        evaluate_population(offspring, test_function_one)

        population = copy.deepcopy(offspring)

    best_individual = min(population, key=lambda ind: ind.fitness)
    return best_individual, best_fitness_list, mean_fitness_list

# Parameters for the real-valued genetic algorithm
P = 150  # population size
generations = 160  # number of generations
tournament_size = 30  # tournament size for selection
crossover_rate = 8  # probability of crossover
mutation_rate = 0.03  # probability of mutation
MUTSTEP = 1  # Size of alterations made during mutation
N = 20  # number of genes in each individual
MIN = -10  # Minimum value for genes
MAX = 10  # Maximum value for genes

# Runs the genetic algorithm for a single run
best_solution, best_fitness_list, mean_fitness_list = genetic_algorithm(P, N, generations, tournament_size, crossover_rate, mutation_rate, MUTSTEP, MIN, MAX)

# Prints the best solution and fitness
print("Best solution:", best_solution.gene)
print("Best fitness:", best_solution.fitness)
print("Mean fitness:", mean_fitness_list[-1])

# Plots the results on the graph
plt.figure(figsize=(10, 6))
plt.plot(range(generations), best_fitness_list, label='Best Fitness')
plt.plot(range(generations), mean_fitness_list, label='Mean Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Best/Mean fitness graph over generations:')
plt.legend()
plt.show()
