from Dynamics import build_dynamics_functions
from scipy.linalg import solve_discrete_are
import casadi as ca
import numpy as np
from Data_generation import generate_dataset # function needed to generate the dataset for calculating
#the polytope W

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

#calculate the polytope W
#we restrict the range that the polytope is calculated for to be under 0.6 rad to better reflect
#the range of values that the controller will encounter in the actual case 
# this is to geneerate new data for the creation of the polytope 
near_eq_conditions = [ 
    [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [-0.1, 0.0], [-0.2, 0.0], [-0.3, 0.0],
    [0.1, 0.2], [0.1, -0.2], [-0.1, 0.2], [-0.1, -0.2],
    [0.2, 0.3], [-0.2, -0.3],
]

near_eq_data = generate_dataset(near_eq_conditions, n_steps=50, u_max=2)

near_eq_states = np.array([entry[0] for entry in near_eq_data])
near_eq_inputs = np.array([entry[1] for entry in near_eq_data])
near_eq_next_states = np.array([entry[2] for entry in near_eq_data])

print(near_eq_states.shape)
print(near_eq_inputs.shape)
print(near_eq_next_states.shape)

