import random
import numpy as np

POPULATION_SIZE = 10
TENTHS = int(POPULATION_SIZE / 10)
OFFSPRINGS_MUL = 5
NUM_OF_SURVIVORS = int(POPULATION_SIZE / OFFSPRINGS_MUL)
MUTATION_CHANCE = 0.4


def create_population(shape: tuple):
    pop_mat = np.random.randint(16, size=shape)
    return pop_mat


def mutation(parents_mat: np.ndarray, pop_mat: np.ndarray):
    probability_mat = np.random.random(parents_mat.shape)
    probability_mat[0:NUM_OF_SURVIVORS, :] = 1  # Keeping parents untouched
    prob_mat_01 = probability_mat < MUTATION_CHANCE
    random_list = create_population((np.count_nonzero(prob_mat_01)))
    pop_mat[probability_mat < MUTATION_CHANCE] = random_list.copy()


def procreate_mat(pop_mat: np.ndarray, parents_ind: np.ndarray):
    repro_mat = np.tile(pop_mat[parents_ind, :], (OFFSPRINGS_MUL, 1))
    return repro_mat


def select_parents_ind(fitness_vect: np.ndarray):
    best_10_indices = np.argsort(fitness_vect)[:TENTHS]
    parents_ind = np.concatenate(
        (best_10_indices, random.choices(range(POPULATION_SIZE), k=TENTHS))
    )
    return parents_ind


def fitness_single(coef_mat: np.ndarray, solution: np.ndarray):
    sol_as_mat = np.tile(solution, (coef_mat.shape[0], 1))
    sol_as_mat[coef_mat != 1] = 0
    cols_sum = np.sum(sol_as_mat, 1)
    fitness = sum(abs(cols_sum - 30))
    return fitness


def fitness_pop(coef_mat: np.ndarray, pop_mat: np.ndarray):
    fitness_vect = [
        fitness_single(coef_mat, pop_mat[row, :]) for row in range(pop_mat.shape[0])
    ]
    return np.asarray(fitness_vect)


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
    pop_mat = create_population((POPULATION_SIZE, 16))
    i = 0
    while True:
        pop_mat[:, 0] = 8
        fitness_vect = fitness_pop(coef_mat, pop_mat)
        arg_min_fit = np.argmin(fitness_vect)
        best_sol = pop_mat[arg_min_fit, :]
        print(
            "Best of generation {i}:\nFitness:{fit:.4f}, Solution Vector: {vect}".format(
                i=i, fit=min_fit, vect=best_sol
            )
        )
        if min_fit == 0:
            break

        parents_ind = select_parents_ind(fitness_vect)
        assert arg_min_fit in parents_ind
        parents_mat = procreate_mat(pop_mat, parents_ind)
        row_exist = [
            parents_mat[row, :] == best_sol for row in range(parents_mat.shape[0])
        ]
        assert np.any(row_exist)
        mutation(parents_mat, pop_mat)
        row_exist = [
            np.all(np.array_equal(pop_mat[row, :], best_sol))
            for row in range(parents_mat.shape[0])
        ]
        assert row_exist
        i += 1


if __name__ == "__main__":
    natural_selection()
