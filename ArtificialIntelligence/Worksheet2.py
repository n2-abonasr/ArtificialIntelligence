import random
import copy
import matplotlib.pyplot as plt

class Individual:
    def __init__(self, N):
        self.gene = [random.randint(0, 1) for _ in range(N)]
        self.fitness = 0

def test_function(ind):
    return sum(ind.gene)

def create_initial_population(P, N):
    return [Individual(N) for _ in range(P)]

def tournament_selection(population, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda ind: ind.fitness)
        selected_parents.append(copy.deepcopy(winner))
    return selected_parents

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

def bitwise_mutation(offspring, P, N, mutation_rate):
    for i in range(P):
        new_ind = Individual(N)
        new_ind.gene = []
        
        for j in range(N):
            gene = offspring[i].gene[j]
            mut_prob = random.random()
            
            if mut_prob < mutation_rate:
                gene = 1 - gene  # flip the bit
                
            new_ind.gene.append(gene)
        
        offspring[i] = copy.deepcopy(new_ind)

def evaluate_population(population, fitness_function):
    for ind in population:
        ind.fitness = fitness_function(ind)

def genetic_algorithm(P, N, generations, tournament_size, crossover_rate, mutation_rate):
    population = create_initial_population(P, N)
    best_fitness_list = []
    mean_fitness_list = []

    for generation in range(generations):
        evaluate_population(population, test_function)

        best_fitness = max(ind.fitness for ind in population)
        mean_fitness = sum(ind.fitness for ind in population) / P

        best_fitness_list.append(best_fitness)
        mean_fitness_list.append(mean_fitness)

        offspring = tournament_selection(population, tournament_size)
        one_point_crossover(offspring, P, N)
        bitwise_mutation(offspring, P, N, mutation_rate)

        evaluate_population(offspring, test_function)

        population = copy.deepcopy(offspring)

    best_individual = max(population, key=lambda ind: ind.fitness)
    return best_individual, best_fitness_list, mean_fitness_list


P = 50  # population size
N = 50  # number of genes in each individual
generations = 50  # number of generations
tournament_size = 2  # tournament size for selection
crossover_rate = 0.8  # probability of crossover
mutation_rate = 0.01  # probability of mutation

# Run the genetic algorithm for a single run
best_solution, best_fitness_list, mean_fitness_list = genetic_algorithm(P, N, generations, tournament_size, crossover_rate, mutation_rate)

print("Best solution:", best_solution.gene)
print("Best fitness:", best_solution.fitness)
print("Mean fitness:", mean_fitness_list[-1])

# Plotting the results
plt.figure(figsize=(10, 6))

plt.plot(range(generations), best_fitness_list, label='Best Fitness')
plt.plot(range(generations), mean_fitness_list, label='Mean Fitness')

plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Best and Mean Fitness over Generations')
plt.legend()
plt.show()
