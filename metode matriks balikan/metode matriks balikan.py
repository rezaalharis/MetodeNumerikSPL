import numpy as np

def inverse_matrix_method(A, b):
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
    return x

# Testing
A = np.array([[2, 1, -1], [3, 4, 2], [1, -5, 3]])
b = np.array([2, 13, 10])
x = inverse_matrix_method(A, b)
print("Solusi menggunakan metode matriks balikan:", x)
print(x)