import random
import numpy as np
import itertools as it

POPULATION_SIZE = 1000
TENTHS = int(POPULATION_SIZE / 10)
OFFSPRINGS_MUL = 5
NUM_OF_SURVIVORS = int(POPULATION_SIZE / OFFSPRINGS_MUL)
MUTATION_CHANCE = 0.5
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
    probability_mat = np.random.random(pop_mat.shape)
    probability_mat[0:NUM_OF_SURVIVORS, :] = 1  # Keeping parents untouched
    prob_mat_01 = np.nonzero(probability_mat < MUTATION_CHANCE)
    swap_prob_mat_01 = prob_mat_01.copy()
    np.apply_along_axis(rng.shuffle, 1, swap_prob_mat_01)
    pop_mat[prob_mat_01] = pop_mat[swap_prob_mat_01]


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
    best_fit = float("inf")
    while True:
        # pop_mat[:, RECT_VECTOR_OFFSET] = 8
        fitness_pop(coef_mat, pop_mat)
        arg_min_fit = np.argmin(pop_mat[:, 0])
        min_fit = pop_mat[arg_min_fit, 0]
        best_sol = pop_mat[arg_min_fit, RECT_VECTOR_OFFSET:]
        if min_fit < best_fit or not i % 200:
            print(
                "Best of generation {i}:\nFitness:{fit:.4f}, Solution Vector: {vect}".format(
                    i=i, fit=min_fit, vect=best_sol
                )
            )
            best_fit = min_fit
        if min_fit == 0:
            break

        pop_mat = keep_only_parents(pop_mat)
        pop_mat = procreate_mat(pop_mat)
        mutation(pop_mat)
        # if i == 1000:
        # pop_mat = create_population(POP_MAT_SHAPE)
        i += 1


if __name__ == "__main__":
    natural_selection()
