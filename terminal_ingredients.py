from Dynamics import build_dynamics_functions
from scipy.linalg import solve_discrete_are
import casadi as ca
import numpy as np

L_TRUE = 0.5
discrepancy_levels = [0.05, 0.15, 0.30, 0.50, 0.75]

Q = np.eye(2)
R = np.array([[1.0]])

x_eq = np.array([0.0, 0.0])
u_eq = np.array([0.0])

terminal_ingredients = {}

for level in discrepancy_levels:
    l_wrong = L_TRUE * (1 - level)
    f_wrong, F_wrong, x, u = build_dynamics_functions(l=l_wrong)

    A_sym = ca.jacobian(f_wrong(x, u), x)
    B_sym = ca.jacobian(f_wrong(x, u), u)
    jac_func = ca.Function('jacobian_func', [x, u], [A_sym, B_sym])

    A, B = jac_func(x_eq, u_eq)
    A = np.array(A)
    B = np.array(B)

    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    eigenvalues = np.linalg.eigvals(A + B @ K)
    is_stable = np.all(np.abs(eigenvalues) < 1)

    terminal_ingredients[level] = {
        'A': A, 'B': B, 'P': P, 'K': K, 'stable': is_stable
    }

    print(f"Discrepancy {level}: stable = {is_stable}, eigenvalues = {eigenvalues}")