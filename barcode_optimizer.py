import numpy as np
import random
import pygad
import os
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from data_loader import load_data
from tqdm import tqdm
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def barcode_optimizer(dict_labels, sel_features, save_file_link_p1, X_train, y_train, X_test, y_test, X_val, y_val):
    """    X_train, y_train = load_data(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels,
                                 fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
    X_test, y_test = load_data(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels,
                               fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
    X_val, y_val = load_data(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels,
                             fl_p1="AllDensePatches", fl_p2="DN121_features_dict")"""

    if sel_features:
        X_train = X_train[:, sel_features]
        X_test = X_test[:, sel_features]
        X_val = X_val[:, sel_features]

    print(X_train.shape, X_test.shape, X_val.shape)

    def generate_binary_barcodes(features):
        train_codes_binary = np.zeros((features.shape[0], features.shape[1] - 1))
        for ir in range(features.shape[0]):
            for ic in range(features.shape[1] - 1):
                if features[ir, ic] <= features[ir, ic + 1]:
                    train_codes_binary[ir, ic] = 1
        return train_codes_binary

    K = 11

    def fitness_train(ga_instance, solution, solution_idx=1):
        order = solution.astype(int)
        new_train_codes = np.copy(X_train[:, order])

        train_codes_binary = generate_binary_barcodes(new_train_codes)

        clf = KNeighborsClassifier(n_neighbors=K)
        clf.fit(train_codes_binary, y_train)
        y_pred = clf.predict(train_codes_binary)

        fitness = f1_score(y_train, y_pred, average='macro')

        # if solution_idx % 10 == 0:
        #    print(f"Solution {solution_idx}: Train F1-score={fitness:.3f}, Test F1-score={fitness_test(solution):.3f}, Validation F1-score={fitness_val(solution):.3f}")
        return fitness * 1

    def fitness_test(solution, solution_idx=1):
        order = solution.astype(int)
        new_train_codes = np.copy(X_train[:, order])
        new_test_codes = np.copy(X_test[:, order])

        train_codes_binary = generate_binary_barcodes(new_train_codes)
        test_codes_binary = generate_binary_barcodes(new_test_codes)

        clf = KNeighborsClassifier(n_neighbors=K)
        clf.fit(train_codes_binary, y_train)
        y_pred = clf.predict(test_codes_binary)

        fitness = f1_score(y_test, y_pred, average='macro')
        # if solution_idx % 10 == 0:
        #    print(f"Solution {solution_idx}: Train F1-score={fitness_train(solution):.3f}, Test F1-score={fitness:.3f}, Validation F1-score={fitness_val(solution):.3f}")
        return fitness * 1

    def fitness_val(solution, solution_idx=1):
        order = solution.astype(int)
        new_train_codes = np.copy(X_train[:, order])
        new_val_codes = np.copy(X_val[:, order])

        train_codes_binary = generate_binary_barcodes(new_train_codes)
        val_codes_binary = generate_binary_barcodes(new_val_codes)

        clf = KNeighborsClassifier(n_neighbors=K)
        clf.fit(train_codes_binary, y_train)
        y_pred = clf.predict(val_codes_binary)

        fitness = f1_score(y_val, y_pred, average='macro')
        return fitness * 1

    def crossover_func(parents, offspring_size, ga_instance):
        offspring = []
        idx = 0
        while len(offspring) < offspring_size[0]:
            ind1 = parents[idx % parents.shape[0], :].copy()
            ind2 = parents[(idx + 1) % parents.shape[0], :].copy()

            size = min(len(ind1), len(ind2))
            a, b = random.sample(range(size), 2)
            if a > b:
                a, b = b, a

            holes1, holes2 = [True] * size, [True] * size
            for i in range(size):
                if i < a or i > b:
                    holes1[ind2[i]] = False
                    holes2[ind1[i]] = False

            # We must keep the original values somewhere before scrambling everything
            temp1, temp2 = ind1, ind2
            k1, k2 = b + 1, b + 1
            for i in range(size):
                if not holes1[temp1[(i + b + 1) % size]]:
                    ind1[k1 % size] = temp1[(i + b + 1) % size]
                    k1 += 1

                if not holes2[temp2[(i + b + 1) % size]]:
                    ind2[k2 % size] = temp2[(i + b + 1) % size]
                    k2 += 1

            # Swap the content between a and b (included)
            for i in range(a, b + 1):
                ind1[i], ind2[i] = ind2[i], ind1[i]

            offspring.append(ind1)
            offspring.append(ind2)

            idx += 1
        return np.array(offspring)[:offspring_size[0]]

    fitness_function = fitness_train

    dimensions = X_train.shape[1]
    print(f"Number of Features: {dimensions}")
    num_generations = int(min(np.power(2, dimensions), 1000))
    num_population = int(min(np.power(2, dimensions), 100))
    num_parents_mating = int(num_population * 0.2)

    init_range_low = 0
    init_range_high = dimensions - 1

    parent_selection_type = "random"
    keep_parents = 1

    crossover_type = crossover_func

    mutation_type = "swap"

    num_runs = 5

    res = [["K", "Train F1", "Test F1", "Validation F1"]]
    # %%
    # K=11
    sol = np.linspace(init_range_low, init_range_high, dimensions)
    # print(fitness_train(None,sol,0))
    # print(fitness_test(sol))
    # print(fitness_val(sol))
    res.append([K, fitness_train(None, sol, 0), fitness_test(sol), fitness_val(sol)])
    df = pd.DataFrame(res)
    print(df.head())

    def on_generation(ga):
        print(
            f"Generation {ga.generations_completed}. Train F1-score: {ga.best_solutions_fitness[-1]:.3f}, Test F1-score: {fitness_test(ga.best_solutions[-1]):.3f}, Validation F1-score: {fitness_val(ga.best_solutions[-1]):.3f}")

    def initialize_pop():
        # new_population = np.random.uniform(low=init_range_low, high=init_range_high+1, size=[sol_per_pop, num_genes])
        original_sol = np.linspace(init_range_low, init_range_high, dimensions)
        initial_population = []
        initial_population.append(original_sol)
        for i in range(num_population - 1):
            initial_population.append(np.random.permutation(original_sol))

        initial_population = np.array(initial_population).astype(int)
        return initial_population

    save_file_link = f"{save_file_link_p1}_MaxGens{num_generations}_NP{num_population}_K{K}_Runs{num_runs}.npz"

    if not os.path.isfile(save_file_link):
        np.savez(save_file_link,
                 last_population=[],
                 best_solutions=[],
                 bf_train_per_gen=[],
                 bf_val_per_gen=[],
                 bf_test_per_gen=[])
        start_run = 0
    else:
        npload = np.load(save_file_link)
        start_run = npload['best_solutions'].shape[0]

    for run in range(start_run, num_runs):
        print("Run ", run)

        population = initialize_pop()
        ga = pygad.GA(num_generations=num_generations - 1,
                      num_parents_mating=num_parents_mating,
                      fitness_func=fitness_train,
                      sol_per_pop=num_population,
                      num_genes=dimensions,
                      initial_population=population,
                      parent_selection_type='rank',
                      keep_elitism=1,
                      mutation_type=mutation_type,
                      mutation_probability=0.1,
                      crossover_type=crossover_type,
                      on_generation=on_generation,
                      gene_type=int,
                      allow_duplicate_genes=False,
                      save_best_solutions=True)
        # run = 1

        ga.run()

        last_pop = ga.population
        allgens_bests = ga.best_solutions
        allgens_bests_train_f1 = ga.best_solutions_fitness

        # %%
        allgens_bests_test_f1, allgens_bests_val_f1 = np.zeros(num_generations), np.zeros(num_generations)
        # %%
        # %%
        for i in tqdm(range(num_generations)):
            allgens_bests_test_f1[i] = fitness_test(allgens_bests[i])
            allgens_bests_val_f1[i] = fitness_val(allgens_bests[i])
        # %%
        npload = np.load(save_file_link)
        best_solutions = npload['best_solutions'].tolist()
        bf_train_per_gen = npload['bf_train_per_gen'].tolist()
        bf_val_per_gen = npload['bf_val_per_gen'].tolist()
        bf_test_per_gen = npload['bf_test_per_gen'].tolist()
        # %%
        bf_val_per_gen.append(allgens_bests_val_f1)
        bf_test_per_gen.append(allgens_bests_test_f1)
        bf_train_per_gen.append(allgens_bests_train_f1)
        best_solutions.append(allgens_bests)
        # %%
        np.savez(save_file_link,
                 last_population=last_pop,
                 best_solutions=np.array(best_solutions),
                 bf_train_per_gen=np.array(bf_train_per_gen),
                 bf_val_per_gen=np.array(bf_val_per_gen),
                 bf_test_per_gen=np.array(bf_test_per_gen))
