import random
import copy

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

def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1.gene) - 1)
    child1_gene = parent1.gene[:crossover_point] + parent2.gene[crossover_point:]
    child2_gene = parent2.gene[:crossover_point] + parent1.gene[crossover_point:]
    child1 = Individual(len(child1_gene))
    child1.gene = child1_gene
    child2 = Individual(len(child2_gene))
    child2.gene = child2_gene
    return child1, child2

def bitwise_mutation(ind, mutation_rate):
    mutated_gene = [bit ^ (random.random() < mutation_rate) for bit in ind.gene]
    mutated_ind = Individual(len(mutated_gene))
    mutated_ind.gene = mutated_gene
    return mutated_ind

def evaluate_population(population, fitness_function):
    for ind in population:
        ind.fitness = fitness_function(ind)

def genetic_algorithm(P, N, generations, tournament_size, crossover_rate, mutation_rate):
    population = create_initial_population(P, N)
    for generation in range(generations):
        evaluate_population(population, test_function)
        offspring = []
        while len(offspring) < P:
            parents = tournament_selection(population, tournament_size)
            if random.random() < crossover_rate:
                child1, child2 = single_point_crossover(parents[0], parents[1])
                offspring.extend([child1, child2])
            else:
                offspring.extend(parents)
        
        offspring = [bitwise_mutation(ind, mutation_rate) for ind in offspring]

        evaluate_population(offspring, test_function)

        if sum(ind.fitness for ind in offspring) > sum(ind.fitness for ind in population):
            population = offspring

    best_individual = max(population, key=lambda ind: ind.fitness)
    return best_individual

# Example 
P = 50  # population size
N = 10  # number of genes in each individual
generations = 100  # number of generations
tournament_size = 2  # tournament size for selection
crossover_rate = 0.8  # probability of crossover
mutation_rate = 0.01  # probability of mutation

best_solution = genetic_algorithm(P, N, generations, tournament_size, crossover_rate, mutation_rate)
print("Best solution:", best_solution.gene)
print("Fitness:", best_solution.fitness)
