import numpy as np


def js(cost_fn, lower_bound, upper_bound, nd, iterations, population, logistic_map=None, termination=True):
    """
    """
    n_var = nd
    var_min = np.array(lower_bound if len(lower_bound) > 1 else lower_bound * np.ones(nd))
    var_max = np.array(upper_bound if len(upper_bound) > 1 else upper_bound * np.ones(nd))

    max_it, n_pop = iterations, population

    # """ added by Paul"""
    # # Ensure population isn't set to 1, as the algorithm won't work
    # if n_pop == 1:
    #     print("Error: Population size cannot be 1 for the Jellyfish Search algorithm. Terminating.")
    #     exit()  # Terminate the script

    if logistic_map is None:
        popi = initialization_constraint(n_pop, nd, var_max, var_min)  # rand generation
        # print(f'Randomly generated logistic logistic_map')

    else:
        popi = logistic_map

    popi = [repair_weights(pop, lower_bound, upper_bound) for pop in popi]

    pop_cost = [cost_fn(popi[i], var_min, var_max) for i in range(n_pop)]

    fbestvl = []

    for it in range(max_it):
        meanvl = np.mean(popi, axis=0)
        index = np.argsort(pop_cost)
        best_sol = popi[index[0]]
        best_cost = pop_cost[index[0]]

        for i in range(n_pop):
            ar = (1 - it / max_it) * (2 * np.random.rand() - 1)

            if abs(ar) >= 0.5:
                newsol = popi[i] + np.random.rand(n_var) * (best_sol - 3 * np.random.rand() * meanvl)
                newsol = repair_weights(newsol, var_min, var_max)
                # print('after js: ->>>', newsol) #pol tets
                newsol_cost = cost_fn(newsol, var_min, var_max)

                if newsol_cost < pop_cost[i]:
                    popi[i] = newsol
                    pop_cost[i] = newsol_cost

                    if pop_cost[i] < best_cost:
                        best_cost = pop_cost[i]
                        best_sol = popi[i]
            else:
                if np.random.rand() <= (1 - ar):
                    j = i
                    while j == i:
                        j = np.random.choice(n_pop)
                    step = popi[i] - popi[j]

                    if pop_cost[j] < pop_cost[i]:
                        step = -step

                    newsol = popi[i] + np.random.rand(n_var) * step
                else:
                    newsol = popi[i] + 0.1 * (var_max - var_min) * np.random.rand()

                newsol = repair_weights(newsol, var_min, var_max)
                # print('newsol swarm movement: ', newsol ) #pol test
                newsol_cost = cost_fn(newsol, var_min, var_max)

                if newsol_cost < pop_cost[i]:
                    popi[i] = newsol
                    pop_cost[i] = newsol_cost

                    if pop_cost[i] < best_cost:
                        best_cost = pop_cost[i]
                        best_sol = popi[i]

        fbestvl.append(best_cost)

        if termination:

            if it >= 500:  # reduce this to a smaller number to hit early stopping more easily
                if abs(fbestvl[it] - fbestvl[it - 300]) < 1e-35:  # increased this for quicker convergence detection
                    # print("Termination condition met at iteration:", it)
                    break

    return best_sol, fbestvl[-1], it * n_pop, fbestvl, it


def main_js(iterations, population, Lb, Ub, nd, map=None, termination=True):
    # Setting up bounds and parameters

    # sharpe:
    function = negative_portfolio_sharpe_ratio

    # Running the JS optimizer
    start_time = time.time()
    best_solution, best_cost, evaluations, cost_over_time, iterations_js = js(function, Lb, Ub, nd, iterations,
                                                                              population, map, termination)
    elapsed_time = time.time() - start_time

    # display_metrics(function, best_solution, best_cost, iterations_js, evaluations, elapsed_time, cost_over_time,
    # 'JS')
    return best_solution, best_cost, iterations_js, evaluations, elapsed_time, cost_over_time
