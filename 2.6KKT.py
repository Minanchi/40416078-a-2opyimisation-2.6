import numpy as np
from scipy.optimize import minimize

# Given matrices and vectors
B = np.random.uniform(-1, 1, (5, 5))
Q = np.dot(B.T, B) + np.eye(5)
A = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0]]
)
b = np.array([1, -1])

# Define the Lagrangian function
def lagrangian(x, lambda_):
    return 0.5 * x.T @ Q @ x + 0.15 * np.sum(x) - lambda_.T @ (A @ x - b)

# Define the KKT system equation
def kkt_equations(vars):
    x, lambda_ = np.split(vars, [5])
    g1 = A @ x - b
    g2 = lambda_.T @ g1
    dL_dx = Q @ x + 0.15 * np.ones(5) - A.T @ lambda_
    dL_dlambda = g1
    return np.concatenate((dL_dx, np.array([g2]), dL_dlambda))  # 修改此处

# Initial guesses for x and lambda
initial_x = np.zeros(5)
initial_lambda = np.zeros(2)

# Combined with an initial guess
initial_vars = np.concatenate((initial_x, initial_lambda))

# Solving the KKT system using a solver
result = minimize(lambda vars: np.sum(kkt_equations(vars) ** 2), initial_vars, method='SLSQP')

# Get optimized x and lambda
optimized_x, optimized_lambda = np.split(result.x, [5])

print("Optimized x:", optimized_x)
print("optimal target value:", lagrangian(optimized_x, optimized_lambda))

