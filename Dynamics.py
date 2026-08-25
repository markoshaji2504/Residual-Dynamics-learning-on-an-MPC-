import casadi as ca 
#declare the variables for the pendulum parameters
m = 1.0    # mass, kg
l = 0.5    # length, m
b = 0.1    # damping coefficient
g = 9.81   # gravity, m/s^2
#initilize the symbolic variables 
theta=ca.SX.sym("theta")
theta_dot=ca.SX.sym("theta_dot")
u=ca.SX.sym("u")
x = ca.vertcat(theta, theta_dot)
#set up the system dynamics 
#angular acceleration
theta_ddot = (m * g * l * ca.sin(theta) + u - b * theta_dot) / (m * l**2)
#rate of change of vector 
x_dot = ca.vertcat(theta_dot, theta_ddot)




