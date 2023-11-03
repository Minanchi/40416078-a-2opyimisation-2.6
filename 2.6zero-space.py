import numpy as np
from scipy.optimize import minimize

# Define matrices and vectors
B = np.random.uniform(-1, 1, size=(5, 5))
Q = B.T @ B + np.eye(5)
A = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0]])
b = np.array([1, -1])

# Find the zero space of the matrix A
nullspace = np.linalg.matrix_rank(A)
nullspace_matrix = np.linalg.matrix_rank(np.linalg.pinv(A))

# Define the objective function
def objective_function(x):
    return 0.5 * x.T @ Q @ x + 0.15 * np.sum(x)

# Defining Equality Constraints
def constraint_eq(x):
    return np.dot(A, x) - b

# Generating Initial Guesses Using Zero Space
initial_guess = np.linalg.pinv(A) @ b

# Define the boundaries of x
bounds = [(None, None)] * 5

#Solving Optimization Problems
result = minimize(objective_function, initial_guess, constraints={'type': 'eq', 'fun': constraint_eq}, bounds=bounds)

# Extracting the optimal solution
optimal_x_nullspace = result.x

print("Optimal solutions obtained using the zero-space method:", optimal_x_nullspace)
