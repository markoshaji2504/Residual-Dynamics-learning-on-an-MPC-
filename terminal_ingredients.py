from Dynamics import build_dynamics_functions
import casadi as ca
import numpy as np
from scipy.linalg import solve_discrete_are #necessary for the algebaraic discrete ricatti equation 


L_TRUE = 0.5
DISCREPANCY = 0.30  # start with one level, generalize later


l_wrong = L_TRUE * (1 - DISCREPANCY)
f_wrong, F_wrong ,x,u= build_dynamics_functions(l=l_wrong)
#at ths point we edit the function to return x and u @@@do it now if not already done 
#edited the function to return x and u 


#compute the jacobians for each discrepancy level 


A_sym = ca.jacobian(f_wrong(x, u), x)
B_sym = ca.jacobian(f_wrong(x, u), u)

jac_func = ca.Function('jacobian_function', [x, u], [A_sym, B_sym])

x_eq = np.array([0.0, 0.0])
u_eq = np.array([0.0])

A, B = jac_func(x_eq, u_eq)
A = np.array(A)
B = np.array(B)

print("A =", A)
print("B =", B)

Q = np.eye(2)
R = np.array([[1.0]]) #Q and R necessary for solving the discrete algebaric ricatti equation 

#solve the dac

P = solve_discrete_are(A, B, Q, R)
print("P =", P)

#now calculate gain K 

K = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
print("K =", K)

#check if the computed matrices make the actual system stable?? 
# #confirm the computed K stabilizes the linearized system 
eigenvalues = np.linalg.eigvals(A + B @ K)
print("Eigenvalues of A + BK:", eigenvalues)

# if egien values have all magnitude less than one then the system is stable since it is a linear system 

