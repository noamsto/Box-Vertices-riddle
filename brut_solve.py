import random
import numpy as np

success = 0


def check_solution(mat, solution):
    global success
    # print(np.array([1, 2, 3, 4]).shape)
    # return
    # print("Solution Vector={vec}".format(vec=solution))
    sol_as_mat = np.tile(solution, (mat.shape[0], 1))
    # print("solution matrix is:\n{mat}".format(mat=sol_as_mat))
    sol_as_mat[mat != 1] = 0
    # print("Selected Options Mat:\n{sel_mat}".format(sel_mat=sol_as_mat))
    cols_sum = np.sum(sol_as_mat, 1)
    all_eq30 = cols_sum == 30
    num_of_30 = np.count_nonzero(all_eq30)
    success = max(num_of_30, success)
    if all(all_eq30):
        print("Solution Vector={vec}".format(vec=solution))
        print("Columns sum is: {vec}".format(vec=cols_sum))
        return True
    return False


def read_mat():
    a_mat = np.genfromtxt(
        open("mell_matrix.csv", "rb"),
        delimiter=",",
        filling_values=0,
        usecols=range(0, 16),
    )
    return a_mat


if __name__ == "__main__":
    a_mat = read_mat()
    print("Got this Matrix")
    print(a_mat)

    options = list(range(0, 16))
    options.remove(8)
    i = 0
    while True:
        if not i % 10 ** 6:
            print("Iteration: " + str(i))
            print("Number of rows equal to 30: " + str(success))
        i += 1
        # options.remove(8)
        picked_options = random.choices(options, k=15)
        # random.shuffle(options)
        # options.insert(0, 8)
        picked_options.insert(0, 8)
        options_arr = np.asarray(picked_options)
        if check_solution(a_mat, options_arr):
            break
