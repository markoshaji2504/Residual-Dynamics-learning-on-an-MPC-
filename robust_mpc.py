from Dynamics import build_dynamics_functions
import do_mpc
import casadi as ca
import numpy as np

L_TRUE = 0.5
DISCREPANCY = 0.30  # start with one level

f_true, F_true, x_true, u_true = build_dynamics_functions(l=L_TRUE)

l_wrong = L_TRUE * (1 - DISCREPANCY)
f_wrong, F_wrong, x_wrong, u_wrong = build_dynamics_functions(l=l_wrong)

#build the mpc model incorporating length as an uncertain parameter to aid in robustness 

model = do_mpc.model.Model('discrete')

theta = model.set_variable(var_type='_x', var_name='theta')
theta_dot = model.set_variable(var_type='_x', var_name='theta_dot')
u = model.set_variable(var_type='_u', var_name='u')
l_param = model.set_variable(var_type='_p', var_name='l_param')

#write the dynamics symbolically icnroporating the uncertain parameter of length

m = 1.0
b = 0.1
g = 9.81
dt = 0.02

theta_ddot = (m * g * l_param * ca.sin(theta) + u - b * theta_dot) / (m * l_param**2)

x = ca.vertcat(theta, theta_dot)
x_dot = ca.vertcat(theta_dot, theta_ddot)

# RK4 discretization, same as before, but now l_param flows through symbolically
f_expr = ca.Function('f_expr', [x, u, l_param], [x_dot])

k1 = f_expr(x, u, l_param)
k2 = f_expr(x + dt/2 * k1, u, l_param)
k3 = f_expr(x + dt/2 * k2, u, l_param)
k4 = f_expr(x + dt * k3, u, l_param)
x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

model.set_rhs('theta', x_next[0])
model.set_rhs('theta_dot', x_next[1])
model.setup()

#n_robust is now non zero telling the mpc that there is a parameter we are uncertain about and that is constantly wrong 
mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 20,
    't_step': 0.02,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

w_bound = 0.05  # placeholder margin for now, in length units — we'll tie this to your actual W bound shortly
l_scenarios = np.array([l_wrong, l_wrong * (1 + w_bound), l_wrong * (1 - w_bound)])
mpc.set_uncertainty_values(l_param=l_scenarios)