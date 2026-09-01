#create the logic for the nominal mpc controller 
#import from dynamics file
from Dynamics import F
#import libraries
import do_mpc
import casadi as ca
import numpy as np 
import matplotlib.pyplot as plt
model_type = 'discrete' #using RK4 discritized dynamics 
model = do_mpc.model.Model(model_type)
#set variables for nominal mpc 
theta = model.set_variable(var_type='_x', var_name='theta')
theta_dot = model.set_variable(var_type='_x', var_name='theta_dot')
u = model.set_variable(var_type='_u', var_name='u')

#set up pendulum dynamics for the mpc 
x_next = F(ca.vertcat(theta, theta_dot), u)
model.set_rhs('theta', x_next[0])
model.set_rhs('theta_dot', x_next[1])
model.setup()

#creating the controller 
mpc = do_mpc.controller.MPC(model)

#setting basic parameters 
#mpc with no uncetianty 
setup_mpc = {
    'n_horizon': 20,        # how many steps ahead the MPC plans
    't_step': 0.02,          # must match the dt  used in RK4 discretization
    'n_robust': 0,           # 0 = no robustness yet 
    'store_full_solution': True,
}

mpc.set_param(**setup_mpc)

#setting up cost function 

lterm = theta**2 + theta_dot**2 + 0.1 * u**2
mterm = theta**2 + theta_dot**2

#registering them with the MPC object 

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(u=0.01)

#setting up torque constraints for the motor 
mpc.bounds['lower', '_u', 'u'] = -5
mpc.bounds['upper', '_u', 'u'] = 5

#finalize mpc setup 
mpc.setup()

#simulate 
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=0.02)
simulator.setup()

#closed loop test loop 
x0 = np.array([0.6, 0.2]) # starting angle and angular velocity
mpc.x0 = x0
simulator.x0 = x0
mpc.set_initial_guess()

for k in range(300): #determines how long the model will run for 
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)


#plotting the response seeing if everything works
mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)
#creating figure and axis 
fig, ax = plt.subplots(3, sharex=True, figsize=(8, 6))
#matching plots with setup 
mpc_graphics.add_line(var_type='_x', var_name='theta', axis=ax[0])
mpc_graphics.add_line(var_type='_x', var_name='theta_dot', axis=ax[1])
mpc_graphics.add_line(var_type='_u', var_name='u', axis=ax[2])

ax[0].set_ylabel('theta [rad]')
ax[1].set_ylabel('theta_dot [rad/s]')
ax[2].set_ylabel('u [N*m]')
ax[2].set_xlabel('time [s]')

mpc_graphics.plot_results()
mpc_graphics.reset_axes()
plt.show()