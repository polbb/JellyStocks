import time
from tqdm import tqdm
import utils
import numpy as np
from cost_functions import COST_FUNCTIONS


def initialization_constraint(num_pop, dimension, upper_bound, lower_bound):
    # Constraints were added to the initialization of population (logistic map) to ensure initial pop respect
    # boundaries.

    forbidden_values = {0.0, 0.25, 0.5, 0.75, 1.0}

    # this very the upper/lower bounds - not sure needed here again.
    # if len(lower_bound) == 1:
    #     lower_bound = lower_bound * np.ones(dimension)
    #     upper_bound = upper_bound * np.ones(dimension)

    x = np.random.rand(1, dimension)

    for idx, value in enumerate(x[0]):  # x[0] cause we are accessing the 'only' row of the matrix
        while value in forbidden_values:
            value = np.random.rand()
            x[0][idx] = value

    a = 4

    for i in range(num_pop - 1):
        new_x = a * x[-1] * (1 - x[-1])
        x = np.vstack([x, new_x])

    initial_population = np.zeros((num_pop, dimension))

    for k in range(dimension):
        for i in range(num_pop):
            initial_population[i, k] = lower_bound[k] + x[i, k] * (upper_bound[k] - lower_bound[k])

    return initial_population


def repair_weights(weights, lower_bound, upper_bound):
    n = len(weights)
    lower_bound = np.array(lower_bound if len(lower_bound) > 1 else lower_bound * np.ones(n))
    upper_bound = np.array(upper_bound if len(upper_bound) > 1 else upper_bound * np.ones(n))

    loop_counter = 0

    while True:
        loop_counter += 1
        if loop_counter > 100:
            print("repair_weights stuck in a loop, breaking out after 100 iterations!")
            break

        # Normalize weights
        weights /= np.sum(weights)

        # Adjust out-of-bound weights to the bounds
        weights[weights < lower_bound] = lower_bound[weights < lower_bound]
        weights[weights > upper_bound] = upper_bound[weights > upper_bound]

        # If all weights are in bound, we're done
        if all(lower_bound <= weights) and all(weights <= upper_bound):
            break

        # Redistribute the excess/deficit proportionally among the other weights
        total_excess = np.sum(weights) - 1.0
        in_bound_mask = np.logical_and(lower_bound <= weights, weights <= upper_bound)
        weights[in_bound_mask] -= total_excess * weights[in_bound_mask] / np.sum(weights[in_bound_mask])

    weights = np.around(weights, 4)
    # weights = np.around(weights, 3)

    return weights


class JellyfishOptimizer:
    def __init__(self,
                 num_of_jellyfishes=8,
                 cost_function='sharpe',
                 num_of_iterations=2000,
                 early_termination=False,

                 ):

        self.validate_number_of_jellyfishes(num_of_jellyfishes)
        # Initialize parameters such as risk tolerance, number of iterations, etc.
        self.early_termination = early_termination
        # self.RISK_FREE = risk_free or self.get_risk_free()
        self.number_of_jellyfishes = num_of_jellyfishes
        self.number_of_iterations = num_of_iterations
        self.cost_fn = None

        self.set_cost_function(cost_function)

    def jellyfish_search(self, log_returns, cov_matrix, dimension, lower_bound, upper_bound, risk_free):
        # Implement the optimization logic here
        # Use expected returns and covariance matrix for the optimization
        # Return the optimized portfolio weights
        cost_function = self.cost_fn

        # def js(cost_function, lower_bound, upper_bound, dimension, iterations, population, map=None,
        # termination=True):
        # lower_bound = utils.construct_bounds_if_needed(dimension, lower_bound)
        # upper_bound = utils.construct_bounds_if_needed(dimension, upper_bound)

        max_it, n_pop = self.number_of_iterations, self.number_of_jellyfishes

        jellyfishes = initialization_constraint(n_pop, dimension, upper_bound, lower_bound)

        jellyfishes = [repair_weights(jellyfish, lower_bound, upper_bound) for jellyfish in jellyfishes]
        pop_cost = [cost_function(jellyfish, lower_bound, upper_bound, log_returns, cov_matrix, risk_free
                                  ) for jellyfish in jellyfishes]
        # pop_cost = [cost_function(jellyfishes[i], lower_bound, upper_bound) for i in range(n_pop)]

        convergence = []
        best_sol = None
        it = None

        for it in tqdm(range(max_it), desc="Optimizing", ncols=100):
            meanvl = np.mean(jellyfishes, axis=0)
            index = np.argsort(pop_cost)
            best_sol = jellyfishes[index[0]]
            best_cost = pop_cost[index[0]]

            for i in range(n_pop):
                ar = (1 - it / max_it) * (2 * np.random.rand() - 1)

                if abs(ar) >= 0.5:
                    newsol = jellyfishes[i] + np.random.rand(dimension) * (best_sol - 3 * np.random.rand() * meanvl)
                    newsol = repair_weights(newsol, lower_bound, upper_bound)
                    # print('after js: ->>>', newsol) #pol tets
                    newsol_cost = cost_function(newsol, lower_bound, upper_bound, log_returns, cov_matrix, risk_free)

                    if newsol_cost < pop_cost[i]:
                        jellyfishes[i] = newsol
                        pop_cost[i] = newsol_cost

                        if pop_cost[i] < best_cost:
                            best_cost = pop_cost[i]
                            best_sol = jellyfishes[i]
                else:
                    if np.random.rand() <= (1 - ar):
                        j = i
                        while j == i:
                            # print(f'within the while loop ')
                            j = np.random.choice(n_pop)
                        step = jellyfishes[i] - jellyfishes[j]

                        if pop_cost[j] < pop_cost[i]:
                            step = -step

                        newsol = jellyfishes[i] + np.random.rand(dimension) * step
                    else:
                        newsol = jellyfishes[i] + 0.1 * (upper_bound - lower_bound) * np.random.rand()

                    newsol = repair_weights(newsol, lower_bound, upper_bound)
                    # print('newsol swarm movement: ', newsol ) #pol test
                    newsol_cost = cost_function(newsol, lower_bound, upper_bound, log_returns, cov_matrix, risk_free)

                    if newsol_cost < pop_cost[i]:
                        jellyfishes[i] = newsol
                        pop_cost[i] = newsol_cost

                        if pop_cost[i] < best_cost:
                            best_cost = pop_cost[i]
                            best_sol = jellyfishes[i]

            convergence.append(best_cost)

            if self.early_termination:

                if it >= 500:  # reduce this to a smaller number to hit early stopping more easily
                    if abs(convergence[it] - convergence[it - 300]) < 1e-5:  # increased this for quicker
                        # convergence detection. This value depends on how many decimal points I wan to consider on
                        # sharpe ratio results
                        break
        best_cost = convergence[-1]
        evaluations = it * n_pop
        return best_sol, best_cost, evaluations, convergence, it

    def optimize(self, log_returns, cov_matrix, dimension, lower_bound, upper_bound, risk_free):
        # print(f'printing within optimize in jellyfish_search')
        # Running the JS optimizer
        start_time = time.time()
        best_solution, best_cost, evaluations, cost_over_time, iterations = self.jellyfish_search(log_returns,
                                                                                                  cov_matrix,
                                                                                                  dimension,
                                                                                                  lower_bound,
                                                                                                  upper_bound,
                                                                                                  risk_free)
        elapsed_time = time.time() - start_time

        return best_solution, best_cost, iterations, evaluations, elapsed_time, cost_over_time

    @staticmethod
    def get_risk_free():
        return utils.get_risk_free()

    @staticmethod
    def validate_number_of_jellyfishes(num_of_jellyfishes):
        if not isinstance(num_of_jellyfishes, int):
            raise ValueError('Number of Jellyfishes must be an integer')
        elif num_of_jellyfishes < 2:
            raise ValueError('Minimum number of Jellyfishes is 2')

    def set_cost_function(self, cost_function):
        if cost_function in COST_FUNCTIONS:
            self.cost_fn = COST_FUNCTIONS[cost_function]
        else:
            raise ValueError(f"Unknown cost function name: {cost_function}")
