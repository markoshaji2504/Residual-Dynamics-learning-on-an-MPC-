from Dynamics import build_dynamics_functions
from scipy.linalg import solve_discrete_are
import do_mpc
import casadi as ca
import numpy as np
import torch
from residual_nn import ResidualNN

L_TRUE = 0.5
discrepancy_levels = [0.05, 0.15, 0.30, 0.50, 0.75]

f_true, F_true, x_true, u_true = build_dynamics_functions(l=L_TRUE)

m = 1.0
b = 0.1
g = 9.81
dt = 0.02


def nn_forward_casadi(x_sym, weights, biases):
    """
    Reconstruct the ResidualNN's forward pass symbolically in CasADi.
    x_sym: a CasADi symbolic vector, shape (3,1) -- [theta, theta_dot, u]
    """
    h = x_sym
    for i in range(len(weights) - 1):
        h = ca.mtimes(weights[i], h) + biases[i]
        h = ca.fmax(0, h)  # ReLU

    output = ca.mtimes(weights[-1], h) + biases[-1]
    return output


def run_closed_loop_test(discrepancy, use_nn_correction, n_steps=300, x0_init=None):
    
    l_wrong = L_TRUE * (1 - discrepancy)
    f_wrong_lin, F_wrong_lin, x_lin, u_lin = build_dynamics_functions(l=l_wrong)

    A_sym = ca.jacobian(f_wrong_lin(x_lin, u_lin), x_lin)
    B_sym = ca.jacobian(f_wrong_lin(x_lin, u_lin), u_lin)
    jac_func = ca.Function('jacobian_func', [x_lin, u_lin], [A_sym, B_sym])
    A, B = jac_func(np.array([0.0, 0.0]), np.array([0.0]))
    A, B = np.array(A), np.array(B)

    Q_ric = np.eye(2)
    R_ric = np.array([[1.0]])
    P_level = solve_discrete_are(A, B, Q_ric, R_ric)

    model = do_mpc.model.Model('discrete')
    theta = model.set_variable(var_type='_x', var_name='theta')
    theta_dot = model.set_variable(var_type='_x', var_name='theta_dot')
    u = model.set_variable(var_type='_u', var_name='u')
    l_param = model.set_variable(var_type='_p', var_name='l_param')

    theta_ddot = (m * g * l_param * ca.sin(theta) + u - b * theta_dot) / (m * l_param**2)
    x = ca.vertcat(theta, theta_dot)
    x_dot = ca.vertcat(theta_dot, theta_ddot)
    f_expr = ca.Function('f_expr', [x, u, l_param], [x_dot])

    k1 = f_expr(x, u, l_param)
    k2 = f_expr(x + dt / 2 * k1, u, l_param)
    k3 = f_expr(x + dt / 2 * k2, u, l_param)
    k4 = f_expr(x + dt * k3, u, l_param)
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    if use_nn_correction:
        nn_model = ResidualNN()
        nn_model.load_state_dict(torch.load(f'models/residual_nn_{discrepancy}.pt'))
        nn_model.eval()
        weights, biases = [], []
        for layer in nn_model.net:
            if isinstance(layer, torch.nn.Linear):
                weights.append(layer.weight.detach().numpy())
                biases.append(layer.bias.detach().numpy())

        nn_input = ca.vertcat(theta, theta_dot, u)
        residual_correction = nn_forward_casadi(nn_input, weights, biases)
        x_next = x_next + residual_correction

    model.set_rhs('theta', x_next[0])
    model.set_rhs('theta_dot', x_next[1])
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=20,
        t_step=dt,
        n_robust=1,
        store_full_solution=True,
        nlpsol_opts={'ipopt.print_level': 0, 'print_time': 0},
    )

    w_bound = 0.05
    l_scenarios = np.array([l_wrong, l_wrong * (1 + w_bound), l_wrong * (1 - w_bound)])
    mpc.set_uncertainty_values(l_param=l_scenarios)

    x_vec = ca.vertcat(theta, theta_dot)
    mterm = ca.mtimes([x_vec.T, P_level, x_vec])
    lterm = mterm + 0.1 * u**2
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=0.01)
    mpc.bounds['lower', '_u', 'u'] = -5
    mpc.bounds['upper', '_u', 'u'] = 5
    mpc.setup()

    sim_model = do_mpc.model.Model('discrete')
    theta_s = sim_model.set_variable(var_type='_x', var_name='theta')
    theta_dot_s = sim_model.set_variable(var_type='_x', var_name='theta_dot')
    u_s = sim_model.set_variable(var_type='_u', var_name='u')
    x_s = ca.vertcat(theta_s, theta_dot_s)
    x_next_s = F_true(x_s, u_s)
    sim_model.set_rhs('theta', x_next_s[0])
    sim_model.set_rhs('theta_dot', x_next_s[1])
    sim_model.setup()

    simulator = do_mpc.simulator.Simulator(sim_model)
    simulator.set_param(t_step=dt)
    simulator.setup()

    x0 = x0_init if x0_init is not None else np.array([0.3, 0.0])
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()

    

    print("About to start closed-loop loop...")
    trajectory = [x0.copy()]
    for k in range(n_steps):
        print(f"  starting step {k}")
        u0 = mpc.make_step(x0)
        print(f"  finished step {k}")
        x0 = simulator.make_step(u0).flatten()
        trajectory.append(x0.copy())

    return np.array(trajectory)


if __name__ == "__main__":
    all_results = {}
    for level in discrepancy_levels:
        for use_nn in [False, True]:
            key = (level, use_nn)
            print(f"Running: discrepancy={level}, NN correction={use_nn}")
            traj = run_closed_loop_test(level, use_nn,n_steps=900) #changed to 900 instead of 300 
            all_results[key] = traj
            np.savez(f'results/traj_{level}_{use_nn}.npz', trajectory=traj)

    print("All runs complete.")