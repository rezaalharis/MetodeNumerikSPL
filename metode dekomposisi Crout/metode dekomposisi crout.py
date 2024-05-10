import numpy as np

def lu_decomposition_crout(A, b):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # LU decomposition with Crout method
    for i in range(n):
        U[i, i] = 1
        for j in range(i, n):
            s1 = sum(L[i, k] * U[k, j] for k in range(i))
            L[j, i] = A[j, i] - s1
        for j in range(i+1, n):
            s2 = sum(L[j, k] * U[k, i] for k in range(i))
            U[i, j] = (A[i, j] - s2) / L[i, i]

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
x = lu_decomposition_crout(A, b)
print("Solution using LU decomposition Crout method:", x)
print(x)