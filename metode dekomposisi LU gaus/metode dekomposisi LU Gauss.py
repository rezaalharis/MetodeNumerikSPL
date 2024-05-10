import numpy as np

def lu_decomposition_gauss(A, b):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # LU decomposition with Gauss method
    for i in range(n):
        L[i, i] = 1
        for j in range(i+1):
            s1 = sum(U[k, i] * L[j, k] for k in range(j))
            U[j, i] = A[j, i] - s1
        for j in range(i, n):
            s2 = sum(U[k, i] * L[j, k] for k in range(i))
            L[j, i] = (A[j, i] - s2) / U[i, i]

    # Solve Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))

    # Solve Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i+1, n))) / U[i, i]

    return x

# Testing code
A = np.array([[2, 1, -1], [4, 1, 2], [1, -1, 1]])
b = np.array([1, -2, 0])
x = lu_decomposition_gauss(A, b)
print("Solution using LU decomposition Gauss method:", x)
print(x)