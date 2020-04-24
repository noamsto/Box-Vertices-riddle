import random
import numpy as np
import itertools as it
import copy

POPULATION_SIZE = 1000
TENTHS = int(POPULATION_SIZE / 10)
OFFSPRINGS_MUL = 5
NUM_OF_SURVIVORS = int(POPULATION_SIZE / OFFSPRINGS_MUL)
MUTATION_CHANCE = 0.4
RECT_VECTOR_OFFSET = 1
POP_MAT_SHAPE = (POPULATION_SIZE, 16)


def create_population(shape: tuple):
    rng = np.random.default_rng()
    row = np.arange(shape[1], -1, -1)
    pop_mat = np.tile(row, (shape[0], 1))
    pop_mat[:, [1, 8]] = pop_mat[:, [8, 1]]
    np.apply_along_axis(rng.shuffle, 1, pop_mat[:, 2:])
    return pop_mat


def mutation(pop_mat: np.ndarray):
    rng = np.random.default_rng()
    probability_mat = rng.random(pop_mat.shape)
    probability_mat[0:NUM_OF_SURVIVORS, :] = 1  # Keeping parents untouched
    probability_mat[:, :2] = 1  # Keeping Fitness and 'A' Columns untouched
    cell_to_swap_mat = probability_mat < MUTATION_CHANCE
    rows_swap_indices_mat = [np.nonzero(row)[0] for row in cell_to_swap_mat[:,]]
    shuffle_rows_swap_indices_mat = copy.deepcopy(rows_swap_indices_mat)
    for i, row in enumerate(
        shuffle_rows_swap_indices_mat[NUM_OF_SURVIVORS:], NUM_OF_SURVIVORS
    ):
        rng.shuffle(row)
        pop_mat[i, rows_swap_indices_mat[i]] = pop_mat[i, row]


def procreate_mat(pop_mat: np.ndarray):
    return np.tile(pop_mat, (OFFSPRINGS_MUL, 1))


def keep_only_parents(pop_mat: np.ndarray):
    best_10_indices = np.argsort(pop_mat[:, 0])[:TENTHS]
    parents_ind = np.concatenate(
        (best_10_indices, random.choices(range(POPULATION_SIZE), k=TENTHS))
    )
    return pop_mat[parents_ind, :]


def fitness_single(coef_mat: np.ndarray, solution: np.ndarray):
    sol_as_mat = np.tile(solution, (coef_mat.shape[0], 1))
    sol_as_mat[coef_mat != 1] = 0
    cols_sum = np.sum(sol_as_mat, 1)
    fitness = sum(abs(cols_sum - 30))
    return fitness


def fitness_pop(coef_mat: np.ndarray, pop_mat: np.ndarray):
    fitness_vect = [
        fitness_single(coef_mat, pop_mat[row, RECT_VECTOR_OFFSET:])
        for row in range(pop_mat.shape[0])
    ]
    pop_mat[:, 0] = fitness_vect


def read_mat():
    a_mat = np.genfromtxt(
        open("mell_matrix.csv", "rb"),
        delimiter=",",
        filling_values=0,
        usecols=range(0, 16),
    )
    return a_mat


def natural_selection():
    coef_mat = read_mat()
    pop_mat = create_population(POP_MAT_SHAPE)
    i = 0
    no_improve_counter = 0
    best_fit = float("inf")
    while True:
        # pop_mat[:, RECT_VECTOR_OFFSET] = 8
        fitness_pop(coef_mat, pop_mat)
        arg_min_fit = np.argmin(pop_mat[:, 0])
        min_fit = pop_mat[arg_min_fit, 0]
        best_sol = pop_mat[arg_min_fit, RECT_VECTOR_OFFSET:]
        if min_fit < best_fit or not i % 200:
            no_improve_counter = 0
            print(
                "Best of generation {i}:\nFitness:{fit:.4f}, Solution Vector: {vect}".format(
                    i=i, fit=min_fit, vect=best_sol
                )
            )
            best_fit = min_fit
        else:
            no_improve_counter += 1

        if no_improve_counter >= 100:
            pop_mat = create_population(POP_MAT_SHAPE)
            i = 0
            continue

        if min_fit == 0:
            break

        pop_mat = keep_only_parents(pop_mat)
        pop_mat = procreate_mat(pop_mat)
        mutation(pop_mat)
        i += 1


if __name__ == "__main__":
    natural_selection()
