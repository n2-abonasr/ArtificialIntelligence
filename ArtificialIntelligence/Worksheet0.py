import random

class Solution:
    def __init__(self, N):
        self.variables = [0] * N
        self.utility = 0

def test_function(ind):
    utility = sum(ind.variables)
    return utility

def random_hill_climber(N, loops):
    # Initialize the current solution
    individual = Solution(N)
    for j in range(N):
        individual.variables[j] = random.randint(0, 100)
    individual.utility = test_function(individual)

    for _ in range(loops):
        # Create a new solution as a variation of the current solution
        new_ind = Solution(N)
        new_ind.variables = individual.variables.copy()

        change_point = random.randint(0, N - 1)
        new_ind.variables[change_point] = random.randint(0, 100)
        new_ind.utility = test_function(new_ind)

        # If the new solution is as good or better, update the current solution
        if individual.utility <= new_ind.utility:
            individual.variables[change_point] = new_ind.variables[change_point]
            individual.utility = new_ind.utility

    return individual

# Example usage:
N = 2
LOOPS = 8

final_solution = random_hill_climber(N, LOOPS)
print("Final Solution:", final_solution.variables)
print("Final Utility:", final_solution.utility)
