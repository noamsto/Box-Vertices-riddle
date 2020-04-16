import numpy as np

np.set_printoptions(formatter={"float": "{: 0.1f}".format})

a_mat = np.genfromtxt(
    open("mell_matrix.csv", "rb"),
    delimiter=",",
    filling_values=0,
    usecols=range(0, 17),
)

b_vect = np.array([30] * len(a_mat))
print(a_mat)
print(a_mat.shape)
print(b_vect)

# print(np.sum(a_mat[:, 16]))

# sol_z = np.linalg.solve(a_mat, b_vect)
# print(sol_z)

# triu = np.linalg.inv(a_mat)
# print(triu)

for c in range(a_mat.shape[1]):
    a_mat[abs(a_mat) < 10 ** -5] = 0
    if a_mat[c, c] == 0:  # Check if column is zero
        if not np.count_nonzero(a_mat[c:, c]):
            continue

    # first stage
    # Create a_mat[c,c] = 1
    if a_mat[c, c] != 0:  # Normalize Row by digonal index
        for c2 in range(a_mat.shape[1]):
            a_mat[c, c2] /= a_mat[c, c]
    else:
        for r in range(c + 1, a_mat.shape[0]):
            if a_mat[r, c] != 0:
                break
        a_mat[r, :] /= a_mat[r, c]
        a_mat[c, :] += a_mat[r, :]
    # Second stage
    # Zero all column except a_mat[c, c] using a_mat[c,:]
    for r in range(c + 1, a_mat.shape[0]):
        a_mat[r, :] -= a_mat[c, :].dot(a_mat[r, c])
    print(np.array2string(a_mat, max_line_width=np.inf))
    print("-----")


print(np.array2string(a_mat, max_line_width=np.inf))
print("-----")
sub_a_mat = a_mat[:17, :17]
print(np.array2string(sub_a_mat, max_line_width=np.inf))
print("-----")
# sub_a_mat = np.delete(sub_a_mat, 11, 0)
# print(sub_a_mat)
# print("-----")
# exit()
ia_mat = np.linalg.inv(sub_a_mat)
print(np.array2string(ia_mat, max_line_width=np.inf))
# np.set_printoptions(precision=0)
res_vect = a_mat[:17, -1]
# res_vect = np.delete(res_vect, 11, 0)
# res_vect[0] -= 8
print(res_vect)
res = np.matmul(ia_mat, res_vect)
print(res)
